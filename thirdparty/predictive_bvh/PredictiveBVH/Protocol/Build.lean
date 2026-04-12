-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import Init.Data.FloatArray
import PredictiveBVH.Primitives.Types
import PredictiveBVH.Formulas.Formula
import PredictiveBVH.Spatial.Partition
import PredictiveBVH.Spatial.HilbertRoundtrip

-- ============================================================================
-- 5. MATRIX PARSING  (Float → Int at FFI boundary)
-- ============================================================================

private def floatAt (raw : FloatArray) (i : Nat) : Float :=
  if h : i < raw.size then raw[i]'h else 0.0

-- Convert a metre value to micrometres (signed); used for bounding box coordinates.
private def toUM (f : Float) : Int :=
  if f >= 0.0 then Int.ofNat (f * 1000000.0).toUInt64.toNat
  else -(Int.ofNat ((-f) * 1000000.0).toUInt64.toNat)

-- Convert a m/s velocity to μm/tick (abs, non-negative), ceiling-rounded.
-- Ceiling division ensures the ghost bound is never smaller than the true expansion
-- (C1 VelocityUnderestimate fix: floor would under-estimate → false negatives).
-- Consistent with toUMPerTick2Abs which also uses ceiling.
-- #snippet toUMPerTickAbs
private def toUMPerTickAbs (f : Float) (tickHz : Nat) : Int :=
  let hz := if tickHz == 0 then 60 else tickHz
  let um := (f.abs * 1000000.0).toUInt64.toNat
  Int.ofNat ((um + hz - 1) / hz)
-- #end toUMPerTickAbs

-- Convert a m/s² acceleration to ½·|a|/tickHz² in μm/tick² (abs), ceiling-rounded.
-- Divides by 2·tickHz² so the stored value equals ⌈½·|a_μm_s²| / tickHz²⌉.
-- Ceiling division (n + d - 1) / d ensures estimates are always conservative
-- (never underestimate the expansion), so A_half * k² = exact ½at² term in the
-- polynomial predictiveCostFormula — no division needed in the AST.
private def toUMPerTick2Abs (f : Float) (tickHz : Nat) : Int :=
  let hz    := if tickHz == 0 then 60 else tickHz
  let denom := 2 * hz * hz
  let um    := (f.abs * 1000000.0).toUInt64.toNat
  Int.ofNat ((um + denom - 1) / denom)

-- Parse one 12-column row: [MinX MaxX MinY MaxY MinZ MaxZ Vx Vy Vz Ax Ay Az].
-- Bounds: μm (signed).  Velocity: μm/tick (abs).  Acceleration: μm/tick² (abs).
-- tickHz is the physics tick rate; its effect is baked in here, not in the formulas.
--
-- C1 (VelocityUnderestimate): velocity is ceiling-rounded (toUMPerTickAbs above).
-- C6 (CoordinateFrameMismatch): caller must supply world-frame coordinates; chunk-seam
--   or vehicle-relative offsets must be normalized before calling parseLeaf.
-- C7 (SegmentBoundaryViolation): for decomposed PCVR avatars, caller must supply each
--   segment as a separate row; splitEntityToSegments (below) assists with this.
--
-- Speed-hack clamp (closes G13 at the parser level):
--   Parsed velocity is clamped to vMaxUmPerTick, a per-sim physical ceiling.
--   0 means unclamped (e.g. projectiles with no physics cap).
--   Clamping at parse time bounds ghost box surface area; inflated velocity beyond
--   the cap produces a larger ghost, not a false negative (expansion_covers_k_ticks).
--
-- Deriving vMaxUmPerTick for your sim:
--   vMaxUmPerTick = ceil(maxSpeedMps * 1000000 / tickHz)
--   Human avatar  @ 20 Hz: ceil(10 * 1000000 / 20)  = 500000   (= vMaxPhysical in Types.lean)
--   Helicopter    @ 20 Hz: ceil(80 * 1000000 / 20)  = 4000000
--   Bullet        @ 20 Hz: ceil(900 * 1000000 / 20) = 45000000
--   Human avatar  @ 60 Hz: ceil(10 * 1000000 / 60)  = 166667
private def clampVel (v : Int) (cap : Nat) : Int :=
  if cap == 0 then v  -- 0 = unclamped
  else
    let c := Int.ofNat cap
    if v > c then c else v

/-- Parse one 12-column row from a FloatArray into a LeafData.
    Converts metre/s values to μm/tick (abs, ceiling-rounded) at the FFI boundary.
    Velocity is clamped to vMaxUmPerTick; pass 0 for unclamped (e.g. projectiles). -/
def parseLeaf (raw : FloatArray) (row : Nat) (tickHz : Nat := 60)
    (vMaxUmPerTick : Nat := vMaxPhysical) : LeafData :=
  let b := row * 12
  { entities     := 1,
    bounds       := { minX := toUM (floatAt raw (b+0)), maxX := toUM (floatAt raw (b+1)),
                      minY := toUM (floatAt raw (b+2)), maxY := toUM (floatAt raw (b+3)),
                      minZ := toUM (floatAt raw (b+4)), maxZ := toUM (floatAt raw (b+5)) },
    velocity     := #[clampVel (toUMPerTickAbs (floatAt raw (b+6)) tickHz) vMaxUmPerTick,
                      clampVel (toUMPerTickAbs (floatAt raw (b+7)) tickHz) vMaxUmPerTick,
                      clampVel (toUMPerTickAbs (floatAt raw (b+8)) tickHz) vMaxUmPerTick],
    acceleration := #[toUMPerTick2Abs (floatAt raw (b+9))  tickHz,
                      toUMPerTick2Abs (floatAt raw (b+10)) tickHz,
                      toUMPerTick2Abs (floatAt raw (b+11)) tickHz],
    timeOffset   := 0,
    duration     := 0 }

-- ============================================================================
-- 5a. SEGMENT DECOMPOSITION  (C7 SegmentBoundaryViolation helper)
--
-- For PCVR avatars decomposed into S independent segments (H4 hypothesis), each
-- segment must be registered as a separate BVH leaf.  Passing a single whole-body
-- row risks C7: a fast-moving forearm can violate the ghost bound of the merged box.
--
-- splitEntityToSegments takes one parsed whole-body LeafData and a list of per-
-- segment bounding boxes (caller-supplied from skeleton pose).  Each segment inherits
-- the whole-body velocity and acceleration; per-segment bounds replace the global box.
-- Velocity clamping (clampVel) is already applied by parseLeaf before this call.
-- ============================================================================

/-- Produce one LeafData per segment from a pre-parsed whole-body leaf.
    Each segment inherits whole-body velocity/acceleration; only the bounding box is replaced.
    segBounds must be in world frame (C6 requirement). -/
def splitEntityToSegments (whole : LeafData) (segBounds : List BoundingBox) : List LeafData :=
  segBounds.map fun bb => { whole with bounds := bb }

-- ============================================================================
-- 5b. TRAJECTORY GHOSTING  (recursive time-split BVH)
--
-- A single k-tick AABB expands by A_half·k² on each axis (quadratic).
-- Splitting into two k/2-tick ghosts reduces the acceleration term by 4×.
-- Recursive splitting continues until the split SAH cost ≥ single-box cost.
--
-- Ghost 2's bounds are shifted to the entity's kinematically predicted
-- midpoint position: shift = V·(k/2) + A_half·(k/2)².  Both ghosts keep
-- the same velocity and acceleration (the expansion formula handles the rest).
--
-- Ghosts are inserted as regular leaves; Hilbert sort places each ghost near
-- its actual predicted spatial position, and saturateLoop groups them.
-- The duration field records how many ticks each ghost covers so lbvhAux
-- can call predictiveSAHK with the correct k per leaf.
-- ============================================================================

-- The k-tick expanded bounding box for a leaf (for union / SAH comparisons).
private def ghostExpandedBounds (ld : LeafData) (k : Nat) : BoundingBox :=
  let ki  := Int.ofNat k
  let k2i := Int.ofNat (k * k)
  { ld.bounds with
    maxX := ld.bounds.maxX + ld.velocity[0]! * ki + ld.acceleration[0]! * k2i,
    maxY := ld.bounds.maxY + ld.velocity[1]! * ki + ld.acceleration[1]! * k2i,
    maxZ := ld.bounds.maxZ + ld.velocity[2]! * ki + ld.acceleration[2]! * k2i }

-- Split a leaf into two temporal halves.
-- Ghost 1: current bounds, covers ticks [timeOffset .. timeOffset + half).
-- Ghost 2: bounds shifted to predicted midpoint, covers [timeOffset+half .. timeOffset+k).
private def ghostSplit (ld : LeafData) (k : Nat) : LeafData × LeafData :=
  let half := k / 2
  let hi   := Int.ofNat half
  let h2i  := Int.ofNat (half * half)
  let dx   := ld.velocity[0]! * hi + ld.acceleration[0]! * h2i
  let dy   := ld.velocity[1]! * hi + ld.acceleration[1]! * h2i
  let dz   := ld.velocity[2]! * hi + ld.acceleration[2]! * h2i
  let shifted : BoundingBox :=
    { minX := ld.bounds.minX + dx, maxX := ld.bounds.maxX + dx,
      minY := ld.bounds.minY + dy, maxY := ld.bounds.maxY + dy,
      minZ := ld.bounds.minZ + dz, maxZ := ld.bounds.maxZ + dz }
  ({ ld with duration := half },
   { ld with bounds := shifted, timeOffset := ld.timeOffset + half, duration := half })

-- RDO bonus for keeping a single ghost (not splitting).
-- Mirrors the spatial rdoModeBonus: at high λ, prefer fewer BVH nodes.
-- Formula: λ · SA(single expanded box) · δ / 200
-- (1/200 encodes "1 ghost vs 2 ghosts": (1/1 − 1/2) = 1/2, scaled by /100)
private def ghostRdoBonus (singleSA : Int) (lam delta : Nat) : Int :=
  (lam : Int) * singleSA * (delta : Int) / 200

-- Recursive ghost splitter.  Compares single-box cost vs split cost
-- (two halves + parent traversal overhead + RDO bonus for staying single).
-- At lam=0 this is pure SAH.  At high lam, splitting requires more savings.
-- fuel bounds recursion depth at ≤ k levels (log₂(k) useful levels in practice).
private def recursiveGhostSplitAux (ld : LeafData) (k : Nat) (lam delta : Nat) : Nat → Array LeafData
  | 0        => #[ld]
  | fuel + 1 =>
    let half := k / 2
    if half == 0 then #[ld]
    else
      let (g1, g2)   := ghostSplit ld half
      let singleCost := predictiveSAHK ld (Int.ofNat k)
      let g1Cost     := predictiveSAHK g1 (Int.ofNat half)
      let g2Cost     := predictiveSAHK g2 (Int.ofNat half)
      let parentSA   := surfaceArea (unionBounds (ghostExpandedBounds g1 half)
                                                 (ghostExpandedBounds g2 half))
      let splitCost  := g1Cost + g2Cost + bvhTraversalCost * parentSA
      let bonus      := ghostRdoBonus (surfaceArea (ghostExpandedBounds ld k)) lam delta
      if splitCost + bonus < singleCost then
        recursiveGhostSplitAux g1 half lam delta fuel ++
        recursiveGhostSplitAux g2 half lam delta fuel
      else
        #[ld]

/-- Recursively split a leaf into temporal ghost halves up to depth k.
    At lam=0 this is pure SAH; higher lam favours fewer ghosts via the RDO bonus. -/
def recursiveGhostSplit (ld : LeafData) (k : Nat) (lam delta : Nat := 0) : Array LeafData :=
  recursiveGhostSplitAux ld k lam delta (k + 1)

-- ============================================================================
-- 6. BVH CONSTRUCTION  (LBVH — Hilbert sort + binary-split hierarchy)
-- ============================================================================

-- ── Hilbert code utilities ──────────────────────────────────────────────────

/-- Count leading zeros for a 30-bit code (result ∈ [0, 30]).
    Used to find the common prefix depth of two codes in the Karras split. -/
def clz30 (x : Nat) : Nat :=
  if x == 0 then 30 else 29 - Nat.log2 x

-- Scene bounds: tight union of all leaf AABBs.
private def sceneBoundsOf (leaves : Array LeafData) : BoundingBox :=
  if leaves.isEmpty then { minX := 0, maxX := 1, minY := 0, maxY := 1, minZ := 0, maxZ := 1 }
  else leaves.foldl (fun acc ld => unionBounds acc ld.bounds) leaves[0]!.bounds

/-- 30-bit 3D Hilbert inverse: code → (x, y, z) ∈ [0, 1023]³.
    Proved roundtrip-correct in HilbertRoundtrip.lean. -/
def hilbert3D_inverse (h : Nat) : Nat × Nat × Nat :=
  hilbertInverse h 10

/-- Compute the Hilbert code for a leaf's centroid, normalised to [0, 1023]³
    within the scene AABB. -/
def leafHilbert (ld : LeafData) (scene : BoundingBox) : Nat :=
  let cx := (ld.bounds.minX + ld.bounds.maxX) / 2
  let cy := (ld.bounds.minY + ld.bounds.maxY) / 2
  let cz := (ld.bounds.minZ + ld.bounds.maxZ) / 2
  let sw := max (scene.maxX - scene.minX) 1
  let sh := max (scene.maxY - scene.minY) 1
  let sd := max (scene.maxZ - scene.minZ) 1
  let nx := ((cx - scene.minX) * 1024 / sw).toNat.min 1023
  let ny := ((cy - scene.minY) * 1024 / sh).toNat.min 1023
  let nz := ((cz - scene.minZ) * 1024 / sd).toNat.min 1023
  hilbert3D nx ny nz

-- ── Hilbert cell AABB ───────────────────────────────────────────────────────
--
-- Unlike Morton, Hilbert bit positions do not correspond to fixed axis splits.
-- The spatial region covered by a Hilbert prefix has a complex shape that
-- depends on the curve's rotation state at each level.  We compute the
-- axis-aligned bounding box of the prefix range by inverse-transforming
-- sample codes and taking their coordinate extremes.
--
-- For prefix depth d the range contains 2^(30-d) codes.  We sample up to
-- 1024 evenly spaced codes (exact when stride ≤ 1024, approximate otherwise).
-- The approximation is tight because the Hilbert curve has O(n^{1/3}) cluster
-- diameter — a contiguous range of S codes spans at most S^{1/3} cells per axis.

/-- Convert a normalised grid coordinate (0..1023) to scene-space integer. -/
private def gridToScene (n : Nat) (sceneMin sceneMax : Int) : Int :=
  sceneMin + (n : Int) * (sceneMax - sceneMin) / 1024

/-- Compute the AABB of all Hilbert codes sharing a common prefix with `code`
    at `prefixDepth` bits, mapped back into scene coordinates. -/
def hilbertCellOf (code : Nat) (prefixDepth : Nat) (scene : BoundingBox) : BoundingBox :=
  if prefixDepth == 0 then scene
  else if prefixDepth ≥ 30 then
    -- Single cell: inverse-transform the code directly.
    let (gx, gy, gz) := hilbert3D_inverse code
    { minX := gridToScene gx scene.minX scene.maxX,
      maxX := gridToScene (gx + 1) scene.minX scene.maxX,
      minY := gridToScene gy scene.minY scene.maxY,
      maxY := gridToScene (gy + 1) scene.minY scene.maxY,
      minZ := gridToScene gz scene.minZ scene.maxZ,
      maxZ := gridToScene (gz + 1) scene.minZ scene.maxZ }
  else
    let stride := 1 <<< (30 - prefixDepth)
    let lo := code &&& (~~~(stride - 1))
    -- Sample up to 1024 codes in the prefix range.
    let sampleStep := max (stride / 1024) 1
    let numSamples := (stride + sampleStep - 1) / sampleStep
    let (gx0, gy0, gz0) := hilbert3D_inverse lo
    let init : BoundingBox :=
      { minX := (gx0 : Int), maxX := (gx0 : Int),
        minY := (gy0 : Int), maxY := (gy0 : Int),
        minZ := (gz0 : Int), maxZ := (gz0 : Int) }
    -- Accumulate min/max over sampled inverse coordinates.
    let gridBB := (List.range numSamples).foldl (fun bb i =>
      let h := lo + i * sampleStep
      let (gx, gy, gz) := hilbert3D_inverse h
      { minX := min bb.minX gx, maxX := max bb.maxX gx,
        minY := min bb.minY gy, maxY := max bb.maxY gy,
        minZ := min bb.minZ gz, maxZ := max bb.maxZ gz }) init
    -- Also sample the last code in the range.
    let (lx, ly, lz) := hilbert3D_inverse (lo + stride - 1)
    let gridBB :=
      { minX := min gridBB.minX lx, maxX := max gridBB.maxX lx,
        minY := min gridBB.minY ly, maxY := max gridBB.maxY ly,
        minZ := min gridBB.minZ lz, maxZ := max gridBB.maxZ lz }
    -- Map grid coordinates back to scene space.
    { minX := gridToScene gridBB.minX.toNat scene.minX scene.maxX,
      maxX := gridToScene (gridBB.maxX.toNat + 1) scene.minX scene.maxX,
      minY := gridToScene gridBB.minY.toNat scene.minY scene.maxY,
      maxY := gridToScene (gridBB.maxY.toNat + 1) scene.minY scene.maxY,
      minZ := gridToScene gridBB.minZ.toNat scene.minZ scene.maxZ,
      maxZ := gridToScene (gridBB.maxZ.toNat + 1) scene.minZ scene.maxZ }

-- ── Karras 2012 binary-search split point ───────────────────────────────────
--
-- Finds the rightmost position γ in [first, last-1] such that all codes in
-- [first, γ] share a longer common prefix with codes[first] than codes[last]
-- does.  One recursive halving per step → O(log n) per node.

private def findSplitAux (codes : Array Nat) (fc cp split step last : Nat) : Nat :=
  if step ≤ 1 then split
  else
    let step' := (step + 1) / 2
    let mid   := split + step'
    let split' :=
      if mid < last && clz30 (fc ^^^ codes[mid]!) > cp then mid else split
    findSplitAux codes fc cp split' step' last
termination_by step

private def findSplit (codes : Array Nat) (first last : Nat) : Nat :=
  if first >= last then first
  else
    let fc := codes[first]!
    if fc == codes[last]! then (first + last) / 2          -- equal codes: median
    else findSplitAux codes fc (clz30 (fc ^^^ codes[last]!)) first (last - first) last

-- ── EGraph helpers (must precede lbvhAux) ───────────────────────────────────

/-- Insert a new node and a fresh equivalence class into the E-graph, returning the updated graph and the new class id. -/
def addClass (g : SpatialEGraph) (node : PartitionNode) (cost : Int) (bounds : BoundingBox)
    (firstCode lastCode : Nat) : SpatialEGraph × EClassId :=
  let nodeId  := g.nodes.size
  let classId := g.classes.size
  let cls : EClass :=
    { id := classId, nodes := #[nodeId], minCost := cost,
      bestNode := some nodeId, bounds := bounds,
      firstCode := firstCode, lastCode := lastCode }
  ({ g with nodes   := g.nodes.push node,
            classes := g.classes.push cls },
   classId)

/-- Cost lookup returning `none` for an ID outside the graph's class array. -/
def classCost? (g : SpatialEGraph) (id : EClassId) : Option Int :=
  if h : id < g.classes.size then some g.classes[id].minCost else none

/-- Unwrapping cost lookup; returns 0 for out-of-bounds IDs. Use only when id is known in-bounds. -/
def classCost (g : SpatialEGraph) (id : EClassId) : Int :=
  (classCost? g id).getD 0

/-- Retrieve the entity-tight bounds for an E-graph class; returns default if out of bounds. -/
def classBounds (g : SpatialEGraph) (id : EClassId) : BoundingBox :=
  classBoundsOf g.classes id

-- ── Optimal prediction window (no tuning knobs) ──────────────────────────────
--
-- Objective (both terms in μm²):
--   J(δ) = baseCost / δ  +  totalLeafCost(leaves, tickHz / δ)
-- where baseCost = totalLeafCost(leaves, tickHz)  [tight-bounds cost at δ=1].
--
-- baseCost/δ: amortised rebuild (k=1: rebuild ≈ one tight-bounds traversal).
-- totalLeafCost(δ): traversal cost with δ-tick expanded bounds (grows with δ).
-- Optimal δ* = argmin J(δ) over [1, tickHz] — fast entities pull it down,
-- slow/static entities push it up.  No lam, no external constants.

private def totalLeafCost (leaves : Array LeafData) (k : Nat) : Int :=
  leaves.foldl (fun acc ld => acc + predictiveSAHK ld (Int.ofNat k)) 0

private def optimalDeltaAux (leaves : Array LeafData) (baseCost : Int)
    (bestδ : Nat) (bestJ : Int) (nextδ : Nat) : Nat → Nat
  | 0      => bestδ
  | fuel + 1 =>
    let j     := baseCost / Int.ofNat nextδ + totalLeafCost leaves nextδ
    let (bestδ', bestJ') := if j < bestJ then (nextδ, j) else (bestδ, bestJ)
    optimalDeltaAux leaves baseCost bestδ' bestJ' (nextδ + 1) fuel

/-- Compute the optimal prediction window δ* ∈ [1, tickHz] that minimises
    J(δ) = baseCost/δ + totalLeafCost(leaves, tickHz/δ).
    Returns 1 if leaves is empty or baseCost ≤ 0 (safe default: rebuild every tick). -/
def optimalDelta (leaves : Array LeafData) (tickHz : Nat) : Nat :=
  if leaves.isEmpty || tickHz == 0 then 1
  else
    let baseCost := totalLeafCost leaves 1
    if baseCost ≤ 0 then 1
    else
      -- J(1) = baseCost + totalLeafCost(k=1) = 2·baseCost
      let j1 : Int := baseCost + baseCost
      -- Search δ ∈ [2, tickHz]; fuel = tickHz - 1 steps
      optimalDeltaAux leaves baseCost 1 j1 2 (tickHz - 1)

-- ── C5 (EffectiveDeltaExceeded) RTT precondition ────────────────────────────
-- If the network RTT in ticks exceeds δ*, stale bounding boxes may not cover
-- the entity's actual position when the interest packet arrives.
-- δ_ge_rtt_ticks is the required precondition: configured δ ≥ one-way RTT in ticks.
-- Callers should invoke deltaFromRttTicks to compute the minimum safe δ from
-- measured one-way latency, then pass max(optimalDelta result, deltaFromRttTicks) as δ.

/-- Minimum δ to cover one-way RTT: ceil(rttOnewayMs * tickHz / 1000). -/
def deltaFromRttTicks (rttOnewayMs : Nat) (tickHz : Nat) : Nat :=
  let hz := if tickHz == 0 then 60 else tickHz
  (rttOnewayMs * hz + 999) / 1000  -- ceiling division

/-- Precondition: configured δ must be ≥ one-way RTT in ticks (C5 guard).
    Satisfied when deltaFromRttTicks rttOnewayMs tickHz ≤ configuredDelta. -/
def deltaCoversRtt (configuredDelta rttOnewayMs tickHz : Nat) : Bool :=
  deltaFromRttTicks rttOnewayMs tickHz ≤ configuredDelta

-- ── Recursive Hilbert LBVH (fuel-terminated) ────────────────────────────────
--
-- PartitionNode.parent = Hilbert cell AABB (space-filling volume for RDO).
-- EClass.bounds = entity-tight union of children (SAH quality).
--
-- Each internal node's parentBounds is the AABB of the Hilbert prefix range
-- shared by all leaves in the subtree.  The Hilbert cell is the spatial
-- volume that RDO evaluates when deciding whether to split or keep a single
-- ghost.  Unlike Morton, the Hilbert cell shape varies with the curve's
-- rotation state at each level, so we compute it via hilbertCellOf (which
-- inverse-transforms sampled codes in the prefix range).

-- Effective duration for a leaf: use its own ghost duration if set, else global δ.
private def leafDuration (ld : LeafData) (δ : Nat) : Nat :=
  if ld.duration > 0 then ld.duration else δ

private def lbvhAux (leaves : Array LeafData) (codes : Array Nat) (δ : Nat)
    (first last : Nat) (g : SpatialEGraph)
    : Nat → SpatialEGraph × EClassId
  | 0 =>
      let ld := if h : first < leaves.size then leaves[first] else default
      let fc := if h : first < codes.size  then codes[first] else 0
      addClass g (.none_split ld) (predictiveSAHK ld (leafDuration ld δ)) ld.bounds fc fc
  | fuel + 1 =>
      if first == last then
        let ld := if h : first < leaves.size then leaves[first] else default
        let fc := if h : first < codes.size  then codes[first] else 0
        addClass g (.none_split ld) (predictiveSAHK ld (leafDuration ld δ)) ld.bounds fc fc
      else
        let fc           := if h : first < codes.size then codes[first] else 0
        let lc           := if h : last  < codes.size then codes[last]  else 0
        let depth        := clz30 (fc ^^^ lc)
        let parentBounds := hilbertCellOf fc depth g.scene
        let split        := findSplit codes first last
        let (g1, lId)    := lbvhAux leaves codes δ first       split g  fuel
        let (g2, rId)    := lbvhAux leaves codes δ (split + 1) last  g1 fuel
        -- Entity-tight union for EClass.bounds and SAH cost.
        let ub   := unionBounds g2.classes[lId]!.bounds g2.classes[rId]!.bounds
        -- PartitionNode.parent = Hilbert cell (space-filling volume for RDO).
        -- Axis label is .depth (generic 2-way); Hilbert cells don't have a
        -- fixed axis orientation per level like Morton does.
        let node : PartitionNode := .depth parentBounds lId rId
        let cost := classCost g2 lId + classCost g2 rId + bvhTraversalCost * surfaceArea ub
        addClass g2 node cost ub fc lc

/-- Build the initial Hilbert-sorted LBVH E-graph from a flat FloatArray of entity rows.
    Computes δ* via J(δ) minimisation, ghost-splits high-acceleration leaves, and returns the saturated graph. -/
def buildInitialEGraph (raw : FloatArray) (rows : Nat) (tickHz : Nat := 60)
    (lam delta : Nat := 0) (vMaxUmPerTick : Nat := vMaxPhysical) : SpatialEGraph :=
  if rows == 0 then { nodes := #[], classes := #[], rootId := none, scene := default, optimalDelta := 1 }
  else
    let leaves  := (List.range rows).toArray.map (fun r => parseLeaf raw r tickHz vMaxUmPerTick)
    let δstar   := optimalDelta leaves tickHz
    -- Recursively ghost-split high-acceleration leaves.  Each ghost covers a
    -- shorter window, reducing acceleration-term bloat.  Ghosts are regular
    -- leaves; Hilbert sort places each near its predicted spatial position.
    let allLeaves := leaves.foldl (fun acc ld => acc ++ recursiveGhostSplit ld δstar lam delta) #[]
    -- Recompute scene from ghost leaves so Hilbert normalisation covers all ghosts.
    let scene     := sceneBoundsOf allLeaves
    let pairsList := (allLeaves.map (fun ld => (leafHilbert ld scene, ld))).toList
    let pairs     := (pairsList.mergeSort (fun a b => a.1 ≤ b.1)).toArray
    let sorted := pairs.map Prod.snd
    let codes  := pairs.map Prod.fst
    let g0 : SpatialEGraph :=
      { nodes := #[], classes := #[], rootId := none, scene := scene, optimalDelta := δstar }
    let (g, rootId) :=
      lbvhAux sorted codes δstar 0 (sorted.size - 1) g0 sorted.size
    { g with rootId := some rootId }

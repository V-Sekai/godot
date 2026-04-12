-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import Init.Data.FloatArray
import Init.Data.Array

-- ============================================================================
-- 1. DOMAIN & DATA STRUCTURES
--
-- All spatial coordinates are in MICROMETRES (Int).
-- All velocities are in μm/tick (Int, abs, non-negative after FFI conversion).
-- All accelerations are in μm/tick² (Int, abs, non-negative after FFI conversion).
-- tickHz is a parse-time parameter; the AST never sees it after parseLeaf.
-- All costs are in μm² (Int).
--
-- Converting at the FFI boundary (parseLeaf) keeps the internal algorithm
-- entirely in ℤ, making every comparison decidable and every property
-- provable without Float lemmas or sorry.
-- ============================================================================

-- #snippet domain-aliases
/-- Identifier for an equivalence class in the spatial E-graph. -/
abbrev EClassId := Nat
/-- Identifier for a partition node in the spatial E-graph. -/
abbrev ENodeId  := Nat
-- #end domain-aliases

-- Axis-aligned bounding box; all coordinates in micrometres.
-- #snippet BoundingBox
/-- Axis-aligned bounding box with all coordinates in integer micrometres. -/
structure BoundingBox where
  minX : Int
  maxX : Int
  minY : Int
  maxY : Int
  minZ : Int
  maxZ : Int
  deriving Inhabited, Repr, DecidableEq
-- #end BoundingBox

-- Physics leaf: one entity, all quantities in μm / μm/tick / μm/tick².
-- timeOffset: ticks from "now" when this ghost's window starts (0 = current).
-- duration:   ticks this ghost covers (0 = use the BVH's global δstar).
-- Ghost leaves created by recursiveGhostSplit set both fields; parseLeaf leaves both 0.
-- #snippet LeafData
/-- Physics leaf representing one entity: bounds in μm, velocity in μm/tick, acceleration in μm/tick².
    Ghost leaves set timeOffset and duration; regular parseLeaf leaves leave both at 0. -/
structure LeafData where
  entities     : Nat
  bounds       : BoundingBox
  velocity     : Array Int   -- [|Vx|, |Vy|, |Vz|]  in μm/tick  (abs; = |v_μm_per_s| / tickHz)
  acceleration : Array Int   -- [|Ax|, |Ay|, |Az|]  in μm/tick² (abs; = ⌈½|a_μm_per_s2| / tickHz²⌉)
  timeOffset   : Nat := 0    -- ticks from "now" when this ghost window starts
  duration     : Nat := 0    -- ticks this ghost covers (0 = global δstar)
  deriving Inhabited, Repr
-- #end LeafData

-- #snippet PartitionNode
-- 3D partition node vocabulary.
-- AV1 SPACE-FILLING INVARIANT:
-- Every emitted split node stores `parent : BoundingBox` — the union of entity-tight
-- child bounds at the time the node was created.  Child geometric bounds are derived
-- as exact midpoint halves/quarters/eighths of `parent`.  The stored `parent` is the
-- proof witness used by the space-filling theorems in Section 8d: for any axis-aligned
-- midpoint split, the child regions tile `parent` exactly (manifold + space-filling).
-- Entity-tight bounds remain on leaf EClasses for SAH quality; `parent` enables proofs.
/-- 3D BVH partition node vocabulary covering 2-, 3-, 4-, 8-way splits and leaf.
    Every split node stores a `parent` BoundingBox (the octree cell) as the space-filling proof witness. -/
inductive PartitionNode where
  | none_split (data : LeafData)
  | horz       (parent : BoundingBox) (top bot : EClassId)      -- Y-axis 2-way
  | vert       (parent : BoundingBox) (left right : EClassId)   -- X-axis 2-way
  | depth      (parent : BoundingBox) (front back : EClassId)   -- Z-axis 2-way
  | horz_a     (parent : BoundingBox) (tl tr bot : EClassId)    -- XY T-shape: top split X, bottom full
  | vert_b     (parent : BoundingBox) (left tr br : EClassId)   -- XY T-shape: left full, right split Y
  | xz_a       (parent : BoundingBox) (fl fr back : EClassId)   -- XZ T-shape: front split X, back full
  | horz_4     (parent : BoundingBox) (s1 s2 s3 s4 : EClassId)  -- 4 Y-strips
  | vert_4     (parent : BoundingBox) (s1 s2 s3 s4 : EClassId)  -- 4 X-strips
  | depth_4    (parent : BoundingBox) (s1 s2 s3 s4 : EClassId)  -- 4 Z-strips
  | oct        (parent : BoundingBox) (s1 s2 s3 s4 s5 s6 s7 s8 : EClassId)
  deriving Inhabited, Repr
-- #end PartitionNode

-- ============================================================================
-- 2. E-GRAPH STORAGE
-- ============================================================================

-- #snippet EClass
/-- An equivalence class in the spatial E-graph: tracks the best (lowest-cost) partition node,
    entity-tight bounds for SAH, and Morton-code range for octree cell reconstruction. -/
structure EClass where
  id        : EClassId
  nodes     : Array ENodeId
  minCost   : Int              -- SAH cost in μm²
  bestNode  : Option ENodeId
  bounds    : BoundingBox      -- entity-tight bounds for SAH union computation
  firstCode : Nat              -- Morton code of leftmost (first) entity in this class
  lastCode  : Nat              -- Morton code of rightmost (last) entity in this class
  deriving Inhabited
-- #end EClass

-- #snippet SpatialEGraph
/-- The full spatial E-graph: flat arrays of nodes and classes, the scene AABB,
    the optimal prediction window δ*, and the root class after saturation. -/
structure SpatialEGraph where
  nodes        : Array PartitionNode
  classes      : Array EClass
  rootId       : Option EClassId    -- set by applyRewrites after saturation
  scene        : BoundingBox        -- full scene AABB; used for bottom-up octree cell reconstruction
  optimalDelta : Nat                -- auto-computed optimal prediction window (ticks); 1 = rebuild every tick
  deriving Inhabited
-- #end SpatialEGraph

-- ============================================================================
-- 2b. BOUNDING BOX UTILITIES
-- ============================================================================

-- #snippet surfaceArea
/-- Surface area 2(wh + hd + wd) of a bounding box in μm². Used as the SAH traversal cost proxy. -/
def surfaceArea (b : BoundingBox) : Int :=
  let w := b.maxX - b.minX
  let h := b.maxY - b.minY
  let d := b.maxZ - b.minZ
  2 * (w * h + h * d + w * d)
-- #end surfaceArea

/-- Tight axis-aligned union of two bounding boxes; used for SAH parent computation. -/
def unionBounds (a b : BoundingBox) : BoundingBox :=
  { minX := min a.minX b.minX, maxX := max a.maxX b.maxX,
    minY := min a.minY b.minY, maxY := max a.maxY b.maxY,
    minZ := min a.minZ b.minZ, maxZ := max a.maxZ b.maxZ }

/-- BVH traversal coefficient (dimensionless): 1 means traversal of a parent costs 1 μm² leaf-equivalent. -/
def bvhTraversalCost : Int := 1

/-- The single source of truth for simulation tick rate.
    20 Hz = current default. 10 Hz = hard floor (VRChat IK sync rate; also the
    ≤100 ms Long-Latency-Reflex bound).
    Values below 10 Hz break the mocap-freshness guarantee — do not lower.
    Change this ONE value to retarget the entire pipeline. -/
def simTickHz : Nat := 20

/-- Interest radius: 5 m = 5,000,000 μm. Single source of truth for ghost snap δ
    computation (CodeGen) and adversarial sim (Sim.lean). Internal unit: μm. -/
def interestRadius : Nat := 5000000

/-- Physical velocity ceiling: human sprint ≈ 10 m/s.
    Per-tick value depends on simTickHz: 10 m/s × 1,000,000 μm / tickHz.
    Server-enforced maximum for ghost bound clamping (C1/G13 fix). -/
def vMaxPhysicalAt (tickHz : Nat) : Nat := 10 * 1000000 / (max tickHz 1)

/-- vMaxPhysical at the configured simTickHz. -/
def vMaxPhysical : Nat := vMaxPhysicalAt simTickHz

/-- Hysteresis threshold: simulation steps an entity must remain in the new zone before STAGING begins.
    4 seconds expressed as step count (`simTickHz * 4`). Single source of truth —
    emitted as `pbvh_hysteresis_threshold(hz)` in the generated C/Rust. -/
def hysteresisThreshold : Nat := simTickHz * 4

-- ============================================================================
-- SPACE-FILLING CURVE UTILITIES (for O(N+k) broadphase)
-- ============================================================================

/-- Count leading zeros for a 30-bit space-filling curve code (result ∈ [0, 30]).
    Curve-agnostic: works for both Morton and Hilbert codes. -/
def clz30 (x : Nat) : Nat :=
  if x == 0 then 30 else 29 - Nat.log2 x

-- ── Morton code (retained for BVH cell reconstruction) ──────────────────────

private def expandBits (v : Nat) : Nat :=
  let v := v &&& 0x3FF
  let v := (v ||| (v <<< 16)) &&& 0xFF0000FF
  let v := (v ||| (v <<< 8))  &&& 0x0300F00F
  let v := (v ||| (v <<< 4))  &&& 0x030C30C3
  (v ||| (v <<< 2))            &&& 0x09249249

def morton3D (x y z : Nat) : Nat :=
  expandBits x ||| (expandBits y <<< 1) ||| (expandBits z <<< 2)

-- ── 3D Hilbert curve (Skilling 2004) ────────────────────────────────────────

private def hilbertAxesToTranspose (x0 y0 z0 order : Nat) : Nat × Nat × Nat :=
  let mask := (1 <<< order) - 1
  let x0 := x0 &&& mask
  let y0 := y0 &&& mask
  let z0 := z0 &&& mask
  let (x1, y1, z1) := (List.range (order - 1)).foldl (fun (x, y, z) i =>
    let q := 1 <<< (order - 1 - i)
    let p := q - 1
    let (x, z) := if z &&& q != 0 then (x ^^^ p, z) else
      let t := (x ^^^ z) &&& p; (x ^^^ t, z ^^^ t)
    let (x, y) := if y &&& q != 0 then (x ^^^ p, y) else
      let t := (x ^^^ y) &&& p; (x ^^^ t, y ^^^ t)
    (x, y, z)) (x0, y0, z0)
  let y2 := y1 ^^^ x1
  let z2 := z1 ^^^ y2
  let t := (List.range (order - 1)).foldl (fun t i =>
    let q := 1 <<< (order - 1 - i)
    if z2 &&& q != 0 then t ^^^ (q - 1) else t) 0
  let x3 := x1 ^^^ t
  let y3 := y2 ^^^ t
  let z3 := z2 ^^^ t
  (x3 &&& mask, y3 &&& mask, z3 &&& mask)

private def hilbertTransposeToIndex (x y z order : Nat) : Nat :=
  (List.range order).foldl (fun h bit =>
    let b := order - 1 - bit
    let h := (h <<< 1) ||| ((z >>> b) &&& 1)
    let h := (h <<< 1) ||| ((y >>> b) &&& 1)
    (h <<< 1) ||| ((x >>> b) &&& 1)) 0

/-- Compute a 30-bit 3D Hilbert index from three 10-bit coordinates.
    Better locality than Morton for volume partitioning:
    cluster diameter O(n^{1/3}) vs O(n^{2/3}) (Bader 2013, Ch. 7). -/
def hilbert3D (x y z : Nat) : Nat :=
  let (tx, ty, tz) := hilbertAxesToTranspose x y z 10
  hilbertTransposeToIndex tx ty tz 10



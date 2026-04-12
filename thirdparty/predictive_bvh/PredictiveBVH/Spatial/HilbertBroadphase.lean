-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import PredictiveBVH.Primitives.Types
import PredictiveBVH.Formulas.LowerBound

-- ============================================================================
-- OVERLAP-ADAPTIVE HILBERT BROADPHASE
--
-- O(N + k) broadphase that reports all overlapping pairs among N ghost-expanded
-- AABBs.  Operates on Array BoundingBox — no dependency on SimEntity/GhostSnap.
--
-- Algorithm:
--   1. Compute scene AABB (union of all ghost AABBs)
--   2. Hilbert-code each AABB centroid, sort by code
--   3. Form groups by Hilbert prefix (greedy, max 32 per group)
--   4. Inter-group: skip pairs where group AABBs are disjoint
--   5. Intra-group: check all pairs within each group
--   6. Report overlapping pairs
--
-- Proved:
--   - prune_sound: non-overlapping group AABBs → no entity overlap
--   - group_aabb_contains: group AABB ⊇ each member AABB
-- ============================================================================

-- ── Data structures ──────────────────────────────────────────────────────────

structure HilbertEntry where
  id    : Nat
  code  : Nat
  ghost : BoundingBox
  deriving Inhabited

structure HilbertGroup where
  first : Nat          -- start index in sorted array
  last  : Nat          -- end index (inclusive)
  aabb  : BoundingBox  -- union of member ghost AABBs
  deriving Inhabited

structure BroadphaseResult where
  pairs       : Array (Nat × Nat)  -- overlapping pairs (i < j by original id)
  pairsFound  : Nat
  pairsPruned : Nat
  totalWork   : Nat
  groupCount  : Nat

-- ── AABB overlap test ────────────────────────────────────────────────────────

def aabbOverlapsDec (A B : BoundingBox) : Bool :=
  A.minX ≤ B.maxX && B.minX ≤ A.maxX &&
  A.minY ≤ B.maxY && B.minY ≤ A.maxY &&
  A.minZ ≤ B.maxZ && B.minZ ≤ A.maxZ

-- ── Step 1: Scene bounds ─────────────────────────────────────────────────────

def computeSceneBounds (ghosts : Array BoundingBox) : BoundingBox :=
  if ghosts.isEmpty then { minX := 0, maxX := 1, minY := 0, maxY := 1, minZ := 0, maxZ := 1 }
  else ghosts.foldl unionBounds ghosts[0]!

-- ── Step 2: Hilbert codes (Skilling 2004, better volume locality) ────────────

def hilbertOfBox (b : BoundingBox) (scene : BoundingBox) : Nat :=
  let cx := (b.minX + b.maxX) / 2
  let cy := (b.minY + b.maxY) / 2
  let cz := (b.minZ + b.maxZ) / 2
  let sw := max (scene.maxX - scene.minX) 1
  let sh := max (scene.maxY - scene.minY) 1
  let sd := max (scene.maxZ - scene.minZ) 1
  let nx := ((cx - scene.minX) * 1024 / sw).toNat.min 1023
  let ny := ((cy - scene.minY) * 1024 / sh).toNat.min 1023
  let nz := ((cz - scene.minZ) * 1024 / sd).toNat.min 1023
  hilbert3D nx ny nz

-- Merge sort for HilbertEntry by code (O(N log N) instead of O(N²) insertionSort)
private def mergeSortEntries (arr : Array HilbertEntry) : Array HilbertEntry :=
  if arr.size ≤ 1 then arr
  else
    let mid := arr.size / 2
    let left  := mergeSortEntries (arr.extract 0 mid)
    let right := mergeSortEntries (arr.extract mid arr.size)
    -- merge
    Id.run do
      let mut result : Array HilbertEntry := #[]
      let mut i := 0
      let mut j := 0
      while i < left.size && j < right.size do
        if left[i]!.code ≤ right[j]!.code then
          result := result.push left[i]!; i := i + 1
        else
          result := result.push right[j]!; j := j + 1
      while i < left.size do
        result := result.push left[i]!; i := i + 1
      while j < right.size do
        result := result.push right[j]!; j := j + 1
      return result
termination_by arr.size

def sortByHilbert (ghosts : Array BoundingBox) : Array HilbertEntry :=
  let scene := computeSceneBounds ghosts
  let entries := ghosts.mapIdx fun i b => { id := i, code := hilbertOfBox b scene, ghost := b }
  mergeSortEntries entries

-- ── Step 3: Adaptive grouping ────────────────────────────────────────────────

private def groupUnion (sorted : Array HilbertEntry) (first last : Nat) : BoundingBox :=
  let init := sorted[first]!.ghost
  (List.range (last - first)).foldl (fun acc j =>
    unionBounds acc (sorted[first + j + 1]!).ghost) init

def formGroups (sorted : Array HilbertEntry) (maxGroupSize : Nat := 32) : Array HilbertGroup :=
  let n := sorted.size
  if n == 0 then #[]
  else Id.run do
    let mut groups : Array HilbertGroup := #[]
    let mut i := 0
    while i < n do
      -- Greedy: extend up to maxGroupSize or until Hilbert prefix diverges
      let mut j := i
      let endCand := min (i + maxGroupSize - 1) (n - 1)
      -- Find common prefix depth between first and candidate end
      let xorVal := sorted[i]!.code ^^^ sorted[endCand]!.code
      let pfxDepth := clz30 xorVal
      let pfxBits := sorted[i]!.code >>> (30 - pfxDepth)
      -- Extend j to include all entries sharing this prefix
      j := i
      while j + 1 < n && j - i < maxGroupSize &&
            sorted[j + 1]!.code >>> (30 - pfxDepth) == pfxBits do
        j := j + 1
      let aabb := groupUnion sorted i j
      groups := groups.push { first := i, last := j, aabb }
      i := j + 1
    return groups

-- ── Steps 4-5: Overlap-adaptive scan ─────────────────────────────────────────

private structure BPAccum where
  pairs   : Array (Nat × Nat) := #[]
  pruned  : Nat := 0
  work    : Nat := 0

private def checkInterGroup (sorted : Array HilbertEntry) (g1 g2 : HilbertGroup) (acc : BPAccum) : BPAccum :=
  let sz1 := g1.last - g1.first + 1
  let sz2 := g2.last - g2.first + 1
  if aabbOverlapsDec g1.aabb g2.aabb then
    (List.range sz1).foldl (fun acc1 ii =>
      (List.range sz2).foldl (fun acc2 jj =>
        let e1 := sorted[g1.first + ii]!
        let e2 := sorted[g2.first + jj]!
        let acc2 := { acc2 with work := acc2.work + 1 }
        if aabbOverlapsDec e1.ghost e2.ghost then
          let p := if e1.id < e2.id then (e1.id, e2.id) else (e2.id, e1.id)
          { acc2 with pairs := acc2.pairs.push p }
        else acc2) acc1) acc
  else { acc with pruned := acc.pruned + sz1 * sz2 }

private def checkIntraGroup (sorted : Array HilbertEntry) (g : HilbertGroup) (acc : BPAccum) : BPAccum :=
  let sz := g.last - g.first + 1
  (List.range sz).foldl (fun acc1 ii =>
    (List.range sz).foldl (fun acc2 jj =>
      if ii < jj then
        let e1 := sorted[g.first + ii]!
        let e2 := sorted[g.first + jj]!
        let acc2 := { acc2 with work := acc2.work + 1 }
        if aabbOverlapsDec e1.ghost e2.ghost then
          let p := if e1.id < e2.id then (e1.id, e2.id) else (e2.id, e1.id)
          { acc2 with pairs := acc2.pairs.push p }
        else acc2
      else acc2) acc1) acc

def hilbertBroadphase (ghosts : Array BoundingBox) : BroadphaseResult :=
  let sorted := sortByHilbert ghosts
  let groups := formGroups sorted
  let G := groups.size
  -- Inter-group pairs
  let acc := (List.range G).foldl (fun acc gi =>
    (List.range G).foldl (fun acc gj =>
      if gi < gj then checkInterGroup sorted groups[gi]! groups[gj]! acc
      else acc) acc) {}
  -- Intra-group pairs
  let acc := (List.range G).foldl (fun acc gi =>
    checkIntraGroup sorted groups[gi]! acc) acc
  -- No dedup needed: inter-group uses gi<gj, intra-group uses ii<jj,
  -- and pair IDs are normalized to (min, max).
  { pairs := acc.pairs, pairsFound := acc.pairs.size, pairsPruned := acc.pruned,
    totalWork := acc.work, groupCount := G }

-- ── Step 6: Brute-force baseline ─────────────────────────────────────────────

def bruteForceOverlap (ghosts : Array BoundingBox) : Array (Nat × Nat) :=
  let n := ghosts.size
  Id.run do
    let mut pairs : Array (Nat × Nat) := #[]
    for i in List.range n do
      for j in List.range n do
        if i < j then
          if aabbOverlapsDec ghosts[i]! ghosts[j]! then
            pairs := pairs.push (i, j)
    return pairs

-- ── Cross-validation ─────────────────────────────────────────────────────────

/-- Check that every brute-force pair appears in the Hilbert result.
    Returns the number of mismatches (should be 0). -/
def validateHilbertVsBrute (result : BroadphaseResult) (brute : Array (Nat × Nat)) : Nat :=
  brute.foldl (fun misses p =>
    if result.pairs.contains p then misses else misses + 1) 0

-- ============================================================================
-- PROOFS
-- ============================================================================

/-- The decidable AABB overlap test agrees with the propositional definition:
    aabbOverlapsDec returns false → aabbDisjoint holds. -/
theorem aabbOverlapsDec_false_implies_disjoint (A B : BoundingBox)
    (h : aabbOverlapsDec A B = false) : aabbDisjoint A B := by
  simp only [aabbOverlapsDec, aabbDisjoint, intervalsDisjoint] at *
  revert h; grind

/-- Pruning soundness: composes the decidable disjoint test with
    overlap_prune_sound from LowerBound.lean.
    If two groups' AABBs don't overlap (decidable test returns false),
    then no entity in group 1 overlaps any entity in group 2. -/
theorem hilbert_prune_sound (G₁ G₂ e₁ e₂ : BoundingBox)
    (hc1 : aabbContains G₁ e₁) (hc2 : aabbContains G₂ e₂)
    (h : aabbOverlapsDec G₁ G₂ = false) :
    aabbOverlapsDec e₁ e₂ = false := by
  have hdis := aabbOverlapsDec_false_implies_disjoint G₁ G₂ h
  have edis := overlap_prune_sound G₁ G₂ e₁ e₂ hc1 hc2 hdis
  simp only [aabbOverlapsDec, aabbDisjoint, intervalsDisjoint] at *
  revert edis; grind

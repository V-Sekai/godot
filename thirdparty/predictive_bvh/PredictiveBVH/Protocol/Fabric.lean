-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import PredictiveBVH.Primitives.Types
import PredictiveBVH.Spatial.HilbertBroadphase

-- Import clz30 from Types (moved there for O(N+k) broadphase)
-- clz30 is defined in Types.lean now

-- ============================================================================
-- SHARED-NOTHING FABRIC: Morton-partitioned zones with STAGING migration
--
-- Terminology:
--   Zone  = a single server process owning a spatial partition (= FabricZone in C++)
--   Shard = the entire fabric: all zones, local + networked
--
-- Zone count is DYNAMIC: scales with entity count and spatial distribution.
-- With r128 fixed-point we can handle galaxy-scale coordinates, so zones
-- partition the 30-bit Morton code space adaptively based on entity density.
--
-- Entity-to-zone assignment: 30-bit Morton code prefix (O(1) per entity).
--
-- Migration protocol (CockroachDB-style parallel commits):
--   OWNED(A) → STAGING(A,B) → OWNED(B)
-- During STAGING, both zones hold a valid ghost snap.
-- ============================================================================

/-- Compute optimal zone count based on entity count and target entities per zone.
    Scales from 1 zone (small scenes) to thousands (galaxy-scale).
    Target: 1000-5000 entities per zone for optimal load balancing. -/
def computeOptimalZoneCount (entityCount : Nat) (targetPerZone : Nat := 2500) : Nat :=
  max 1 ((entityCount + targetPerZone - 1) / targetPerZone)

/-- Compute Morton prefix depth for given zone count.
    For N zones, we need ⌈log₂(N)⌉ prefix bits.
    With 30-bit Morton codes, max zones = 2^30 ≈ 1 billion. -/
def zonePrefixDepth (zoneCount : Nat) : Nat :=
  if zoneCount ≤ 1 then 0
  else 30 - clz30 (zoneCount - 1)

/-- Maximum zones supported with 30-bit Morton codes. -/
theorem maxZoneCount : (1 <<< 30) = 1073741824 := by rfl

-- ── Migration state machine ──────────────────────────────────────────────────

-- #snippet MigrationState
inductive MigrationState where
  | owned
  | staging (targetZone : Nat) (arrivalTick : Nat)
  | incoming (fromZone : Nat)
  deriving Inhabited, Repr
-- #end MigrationState

-- ── Fabric latency ───────────────────────────────────────────────────────────

inductive FabricLatency where
  | sameRegion    -- 1 tick
  | crossRegion   -- 4 ticks (NY↔Singapore)
  | satellite     -- 40 ticks (geostationary)
  deriving Inhabited, Repr

def FabricLatency.toTicks : FabricLatency → Nat
  | .sameRegion  => 1
  | .crossRegion => 4
  | .satellite   => 40

-- ── Morton span ──────────────────────────────────────────────────────────────

/-- A contiguous Morton-code interval [lo, hi] (inclusive) representing a zone's
    coverage of the 30-bit Morton space.  Zone i with prefixDepth d owns exactly
    the codes whose top-d bits equal i, i.e. [i·2^(30-d), (i+1)·2^(30-d) − 1].
    Spans are disjoint across zones and together tile all 2^30 codes. -/
structure MortonSpan where
  lo : Nat   -- inclusive lower bound
  hi : Nat   -- inclusive upper bound
  deriving Inhabited, Repr

/-- Number of distinct Morton codes in one zone given prefixDepth prefix bits.
    = 2^(30 − prefixDepth).  For 112 zones (prefixDepth = 7): width = 2^23 ≈ 8M codes. -/
def mortonSpanWidth (prefixDepth : Nat) : Nat :=
  1 <<< (30 - min prefixDepth 30)

/-- Morton span for zone index `zoneIdx` given `prefixDepth` prefix bits. -/
def zoneMortonSpan (zoneIdx prefixDepth : Nat) : MortonSpan :=
  let w := mortonSpanWidth prefixDepth
  { lo := zoneIdx * w
    hi := zoneIdx * w + w - 1 }

/-- Morton span for zone `i` derived from the zones array size.
    Use this to log or inspect which Morton codes zone i owns. -/
def zoneSpan (zones : Array ZoneState) (i : Nat) : MortonSpan :=
  zoneMortonSpan i (zonePrefixDepth zones.size)

/-- True when Morton code `c` falls inside span `s`. -/
def MortonSpan.contains (s : MortonSpan) (c : Nat) : Bool :=
  s.lo ≤ c && c ≤ s.hi

/-- True when two spans share at least one Morton code. -/
def MortonSpan.overlaps (a b : MortonSpan) : Bool :=
  a.lo ≤ b.hi && b.lo ≤ a.hi

-- ── Morton span invariants (general, any prefix depth ≤ 30) ──────────────────

/-- A Morton code `c` lies in zone `i`'s span iff its top `prefixDepth` bits equal `i`.
    This is the operational form used by `formGroups` in MortonBroadphase.lean. -/
theorem span_contains_iff_prefix (c i prefixDepth : Nat) (hd : prefixDepth ≤ 30) :
    (zoneMortonSpan i prefixDepth).contains c = true ↔
      c >>> (30 - prefixDepth) = i := by
  simp only [MortonSpan.contains, zoneMortonSpan, mortonSpanWidth,
             Nat.min_eq_left hd, Bool.and_eq_true, decide_eq_true_eq,
             Nat.shiftRight_eq_div_pow, Nat.shiftLeft_eq, Nat.one_mul]
  generalize hw_eq : (2 : Nat) ^ (30 - prefixDepth) = w
  have hw : 0 < w := by rw [← hw_eq]; exact Nat.two_pow_pos _
  constructor
  · rintro ⟨h1, h2⟩
    apply Nat.div_eq_of_lt_le h1
    rw [Nat.succ_mul]
    omega
  · intro hdiv
    refine ⟨?_, ?_⟩
    · have := Nat.div_mul_le_self c w
      rw [hdiv] at this; exact this
    · have h1 : c < w * (c / w + 1) := Nat.lt_mul_div_succ c hw
      rw [hdiv, Nat.mul_add, Nat.mul_one, Nat.mul_comm] at h1
      omega

/-- Distinct zones have disjoint Morton spans, regardless of prefix depth. -/
theorem zoneSpans_disjoint (i j prefixDepth : Nat) (h : i ≠ j) :
    (zoneMortonSpan i prefixDepth).overlaps (zoneMortonSpan j prefixDepth) = false := by
  simp only [MortonSpan.overlaps, zoneMortonSpan, mortonSpanWidth,
             Bool.and_eq_false_iff, decide_eq_false_iff_not, Nat.not_le,
             Nat.shiftLeft_eq, Nat.one_mul]
  generalize hw_eq : (2 : Nat) ^ (30 - min prefixDepth 30) = w
  have hw : 0 < w := by rw [← hw_eq]; exact Nat.two_pow_pos _
  rcases Nat.lt_or_gt_of_ne h with hlt | hgt
  · right
    have hsucc : (i + 1) * w ≤ j * w := Nat.mul_le_mul_right _ hlt
    rw [Nat.succ_mul] at hsucc
    omega
  · left
    have hsucc : (j + 1) * w ≤ i * w := Nat.mul_le_mul_right _ hgt
    rw [Nat.succ_mul] at hsucc
    omega

-- ── Zone state ───────────────────────────────────────────────────────────────

structure ZoneState where
  id       : Nat
  volume   : BoundingBox          -- spatial region this zone owns
  entities : Array Nat            -- entity indices with authority
  deriving Inhabited

-- ── Zone assignment ───────────────────────────────────────────────────────────

/-- Assign an entity to a zone using 30-bit Morton-code prefix.
    Uses prefix-based assignment: zone = (code >>> (30 - prefixDepth))
    where prefixDepth = ⌈log₂(zoneCount)⌉.
    This scales from 1 zone to 2^30 zones (≈1 billion) for galaxy-scale scenes. -/
def assignToZone (zones : Array ZoneState) (cx cy cz : Int) : Nat :=
  if zones.isEmpty then 0
  else
    let scene := zones[0]!.volume    -- full scene AABB (same on every zone)
    let pt : BoundingBox := { minX := cx, maxX := cx,
                               minY := cy, maxY := cy,
                               minZ := cz, maxZ := cz }
    let code := hilbertOfBox pt scene  -- 30-bit Hilbert code
    let prefixDepth := zonePrefixDepth zones.size
    if prefixDepth == 0 then 0
    else min (zones.size - 1) (code >>> (30 - prefixDepth))

/-- Create N zones, each covering the full scene AABB.
    Zone boundaries are implicit in the Morton-code prefix, not in zone volumes.
    With dynamic zone count, this scales from local scenes to galaxy-scale. -/
def mkZones (scene : BoundingBox) (n : Nat) : Array ZoneState :=
  Array.ofFn (n := n) fun idx =>
    { id := idx.val, volume := scene, entities := #[] }

/-- Create zones with optimal count for given entity count. -/
def mkOptimalZones (scene : BoundingBox) (entityCount : Nat) (targetPerZone : Nat := 2500) : Array ZoneState :=
  let optimalCount := computeOptimalZoneCount entityCount targetPerZone
  mkZones scene optimalCount

/-- assignToZone result is always in [0, zones.size).
    Guarantees no out-of-bounds zone index for any entity position. -/
theorem assignToZone_in_range (zones : Array ZoneState) (cx cy cz : Int)
    (h : 0 < zones.size) : assignToZone zones cx cy cz < zones.size := by
  unfold assignToZone
  simp only [Array.isEmpty_iff]
  split
  · -- zones.size = 0 branch: contradicts h
    omega
  · -- zones non-empty: split on prefixDepth == 0
    split
    · -- prefixDepth == 0: result is 0
      omega
    · -- result is min (zones.size - 1) (...): bounded by zones.size - 1 < zones.size
      exact Nat.lt_of_le_of_lt (Nat.min_le_left _ _) (by omega)

/-- Optimal zone count scales linearly with entity count. -/
theorem optimalZoneCount_scales (n1 n2 : Nat) (h : n1 ≤ n2) :
    computeOptimalZoneCount n1 2500 ≤ computeOptimalZoneCount n2 2500 := by
  unfold computeOptimalZoneCount
  simp [Nat.div_le_div_left]
  omega

/-- With 1 million entities, we get ~400 zones (2500 per zone). -/
theorem millionEntities_zoneCount : computeOptimalZoneCount 1000000 2500 = 400 := by native_decide

/-- With 1 billion entities (galaxy-scale), we get 400,000 zones. -/
theorem billionEntities_zoneCount : computeOptimalZoneCount 1000000000 2500 = 400000 := by native_decide

/-- Zone prefix depth for 400 zones is 9 bits (2^9 = 512 ≥ 400). -/
theorem prefixDepth_400zones : zonePrefixDepth 400 = 9 := by native_decide

/-- Zone prefix depth for 400,000 zones is 19 bits (2^19 = 524288 ≥ 400000). -/
theorem prefixDepth_400kzones : zonePrefixDepth 400000 = 19 := by native_decide

/-- Reassign all entities to zones based on current positions. -/
def reassignEntities (zones : Array ZoneState) (positions : Array (Int × Int × Int)) : Array ZoneState :=
  -- Clear all entity lists
  let empty := zones.map fun z => { z with entities := #[] }
  -- Assign each entity
  (List.range positions.size).foldl (fun acc i =>
    let pos := positions[i]!
    let zi := assignToZone acc pos.1 pos.2.1 pos.2.2
    if zi < acc.size then
      acc.modify zi fun z => { z with entities := z.entities.push i }
    else acc) empty

-- ── STAGING migration protocol ───────────────────────────────────────────────

/-- Per-entity migration tracking. -/
structure EntityMigInfo where
  zone        : Nat              -- current authoritative zone
  migState    : MigrationState   -- owned / staging / incoming
  hysteresis  : Nat := 0         -- ticks entity has been in the new zone region
  deriving Inhabited

/-- Process migrations for one tick.
    Returns updated migration info array + count of new migrations this tick. -/
def processMigrations (migInfo : Array EntityMigInfo) (zones : Array ZoneState)
    (positions : Array (Int × Int × Int)) (latency : FabricLatency)
    (currentTick : Nat) : Array EntityMigInfo × Nat :=
  let lat := latency.toTicks
  (List.range migInfo.size).foldl (fun (acc, newMigs) i =>
    let mi := acc[i]!
    let pos := positions[i]!
    let targetZone := assignToZone zones pos.1 pos.2.1 pos.2.2
    match mi.migState with
    | .owned =>
      if targetZone != mi.zone then
        -- Entity crossed boundary, start hysteresis counter
        let hyst := mi.hysteresis + 1
        if hyst ≥ hysteresisThreshold then
          -- Begin STAGING
          let mi' := { mi with migState := .staging targetZone (currentTick + lat),
                                hysteresis := 0 }
          (acc.set! i mi', newMigs + 1)
        else
          (acc.set! i { mi with hysteresis := hyst }, newMigs)
      else
        (acc.set! i { mi with hysteresis := 0 }, newMigs)
    | .staging target arrivalTick =>
      if currentTick ≥ arrivalTick then
        -- STAGING complete: transfer authority to target zone
        let mi' := { zone := target, migState := .owned, hysteresis := 0 }
        (acc.set! i mi', newMigs)
      else
        (acc, newMigs)
    | .incoming _ =>
      -- Finalize: become owned
      (acc.set! i { mi with migState := .owned }, newMigs)
  ) (migInfo, 0)

/-- Count entities currently in STAGING state. -/
def countInFlight (migInfo : Array EntityMigInfo) : Nat :=
  migInfo.foldl (fun acc mi =>
    match mi.migState with
    | .staging _ _ => acc + 1
    | _ => acc) 0

-- ── Morton AOI band (cross-zone CH_INTEREST relay) ───────────────────────────
--
-- The C++ FabricZone cross-zone relay forwards a neighbor's CH_INTEREST row to
-- local clients iff the entity's 30-bit Morton code is inside a padded band
-- around this zone's own Morton span. The band width is a compile-time knob
-- (AOI_CELLS in C++), expressed in units of `mortonSpanWidth prefixDepth`.
--
-- Two properties matter:
--   1. Coverage: the band always includes this zone's own span, so the relay
--      never drops an entity produced locally.
--   2. Bounded width: total band size is (1 + 2·aoiCells) · cellWidth, which
--      is independent of the global zone count. Per-client fanout therefore
--      stays constant in the fabric size — the superlinear scaling invariant
--      the C++ headers quote at fabric_zone.cpp:816.
--
-- A parallel C++ helper lives at FabricZone::_hilbert_aoi_band.

/-- Morton AOI band for zone `zoneIdx` in a fabric with `prefixDepth` prefix
    bits, padded by `aoiCells` cellWidths on each side. The C++ helper adds
    a `min (2^30 - 1)` clamp on the upper end; here we model the uncapped
    algebraic band so the coverage and width theorems stay clean. The clamp
    is only reachable at the last zone (`zoneIdx = 2^prefixDepth - 1`), where
    the band already covers the top of Morton space. -/
def aoiBand (zoneIdx prefixDepth aoiCells : Nat) : MortonSpan :=
  let w := mortonSpanWidth prefixDepth
  let zoneLo := zoneIdx * w
  let zoneHi := zoneIdx * w + w - 1
  let pad := aoiCells * w
  { lo := zoneLo - min zoneLo pad
    hi := zoneHi + pad }

/-- Coverage: the AOI band always includes every Morton code inside this
    zone's own span. The C++ relay therefore can never drop a locally-
    produced entity regardless of `aoiCells`. -/
theorem aoiBand_covers_self (zoneIdx prefixDepth aoiCells : Nat) :
    let span := zoneMortonSpan zoneIdx prefixDepth
    let band := aoiBand zoneIdx prefixDepth aoiCells
    band.lo ≤ span.lo ∧ span.hi ≤ band.hi := by
  simp only [aoiBand, zoneMortonSpan]
  refine ⟨?_, ?_⟩
  · exact Nat.sub_le _ _
  · omega

/-- Bounded width: the AOI band spans at most `(1 + 2·aoiCells) · cellWidth`
    Morton codes, independent of the global zone count. This is the algebraic
    form of the superlinear-scaling invariant: adding zones shrinks cellWidth
    (the zone-count divisor), so per-client relay bandwidth stays bounded. -/
theorem aoiBand_width_bound (zoneIdx prefixDepth aoiCells : Nat)
    (hw : 0 < mortonSpanWidth prefixDepth) :
    let w := mortonSpanWidth prefixDepth
    let band := aoiBand zoneIdx prefixDepth aoiCells
    band.hi + 1 - band.lo ≤ (1 + 2 * aoiCells) * w := by
  simp only [aoiBand]
  have h1 : (1 + 2 * aoiCells) * mortonSpanWidth prefixDepth =
      mortonSpanWidth prefixDepth + 2 * aoiCells * mortonSpanWidth prefixDepth := by
    rw [Nat.add_mul, Nat.one_mul]
  have h2 : 2 * aoiCells * mortonSpanWidth prefixDepth =
      2 * (aoiCells * mortonSpanWidth prefixDepth) :=
    Nat.mul_assoc 2 aoiCells (mortonSpanWidth prefixDepth)
  rcases Nat.le_total (zoneIdx * mortonSpanWidth prefixDepth)
      (aoiCells * mortonSpanWidth prefixDepth) with hab | hab
  · rw [Nat.min_eq_left hab]
    omega
  · rw [Nat.min_eq_right hab]
    omega

-- ── Fabric stats ─────────────────────────────────────────────────────────────

structure FabricStats where
  migrations     : Nat := 0
  shadowPubs     : Nat := 0
  crossZonePairs : Nat := 0
  inFlightPeak   : Nat := 0
  stagingTicks   : Nat := 0   -- total entity-ticks spent in STAGING

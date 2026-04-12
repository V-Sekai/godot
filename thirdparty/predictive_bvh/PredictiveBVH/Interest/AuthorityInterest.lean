-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import PredictiveBVH.Primitives.Types
import PredictiveBVH.Formulas.Formula
import PredictiveBVH.Protocol.Fabric

-- ============================================================================
-- AUTHORITY / INTEREST SEPARATION
-- (Liu & Theodoropoulos 2014, Ahmed & Shirmohammadi 2009)
--
-- Standard terminology (kept per user preference):
--   zone  = a single server process owning a spatial partition
--   shard = the entire system of zones
--
-- Authority (simulation ownership):
--   The zone that advances an entity's physics state each tick.
--   Exactly one zone holds authority at all times (proved: staging_resolves_to_single_owner).
--   Tracked via EntityMigInfo.zone + MigrationState.
--
-- Interest (area of interest / AOI replica):
--   A read-only ghost copy held by a neighbouring zone.
--   Used to serve CH_INTEREST snapshots to players near the zone boundary
--   without paying the cost of an authority transfer.
--   Ghost count is bounded separately from authority count (InterestCapacity).
--   A zone may hold interest in an entity WITHOUT holding authority.
--
-- Rule (Ahmed 2009, "shared border region"):
--   An entity enters zone B's interest when its k-tick kinematic expansion
--   (ghostBound v a_half k) overlaps zone B's volume.
--   Authority transfers only after hysteresisThreshold ticks of continuous
--   zone-B presence — SEPARATE from interest registration.
--
-- Consequence for capacity:
--   Ghost replicas do NOT consume authority slots.
--   Zone B can hold interest in hundreds of border entities while keeping
--   all MIGRATION_HEADROOM slots free for incoming authority transfers.
-- ============================================================================

-- ── Interest registration threshold ──────────────────────────────────────────

/-- k ticks lookahead for ghost/interest pre-registration (Boulanger 2006).
    An entity whose k-tick expansion reaches zone B's volume is pre-registered
    as an interest replica on zone B before it physically crosses the boundary.
    Using latency ticks so zone B has a full RTT of lookahead. -/
def interestLookahead : Nat := 6  -- latency_ticks (sameRegion default)

/-- Maximum number of interest replicas a zone may hold simultaneously.
    Sized for the border region: entities within interestLookahead ticks of
    the zone boundary, bounded by MIGRATION_HEADROOM to leave authority slots
    free for staging. -/
def InterestCapacity : Nat := 400

-- ── Record types ─────────────────────────────────────────────────────────────

/-- An interest replica: a read-only ghost of an entity whose authority
    remains on a different zone.  Carries enough state to serve
    CH_INTEREST snapshots to nearby players. -/
structure InterestReplica where
  entityId   : Nat     -- stable across authority transfers
  authorZone : Nat     -- zone currently running physics for this entity
  posX       : Int     -- last known position X (μm)
  posY       : Int     -- last known position Y (μm)
  posZ       : Int     -- last known position Z (μm)
  velX       : Int     -- last known velocity X (μm/tick)
  velY       : Int     -- last known velocity Y (μm/tick)
  velZ       : Int     -- last known velocity Z (μm/tick)
  accX       : Int     -- last known acceleration X (μm/tick²)
  accY       : Int     -- last known acceleration Y (μm/tick²)
  accZ       : Int     -- last known acceleration Z (μm/tick²)
  lastTick   : Nat     -- tick at which this replica was last refreshed
  deriving Inhabited, Repr

/-- Zone state extended with an interest replica buffer.
    `entities`  = indices of entities with AUTHORITY on this zone.
    `replicas`  = interest replicas of entities whose authority is elsewhere. -/
structure ZoneStateAI where
  id        : Nat
  volume    : BoundingBox
  entities  : Array Nat          -- authority-holding entity indices
  replicas  : Array InterestReplica  -- interest replicas (no authority)
  deriving Inhabited

-- ── Interest eligibility ──────────────────────────────────────────────────────

/-- True when an entity's k-tick kinematic expansion overlaps zone B's volume
    on a given axis [min, max].  Uses ghostBound from Formula.lean. -/
def overlapsAxis (pos vel acc_half : Int) (k : Nat) (zMin zMax : Int) : Bool :=
  let expand : Int := Int.ofNat (ghostBound vel.toNat acc_half.toNat k)
  pos - expand ≤ zMax && pos + expand ≥ zMin

/-- True when an entity's k-tick expansion box overlaps a zone's bounding box. -/
def inInterestRange (r : InterestReplica) (zVol : BoundingBox) (k : Nat) : Bool :=
  overlapsAxis r.posX r.velX r.accX k zVol.minX zVol.maxX &&
  overlapsAxis r.posY r.velY r.accY k zVol.minY zVol.maxY &&
  overlapsAxis r.posZ r.velZ r.accZ k zVol.minZ zVol.maxZ

-- ── Ghost eviction ───────────────────────────────────────────────────────────

/-- An interest replica expires if not refreshed within 2 × latency ticks.
    The authority zone stops sending ghost updates once entity leaves
    interest range, so the replica naturally ages out. -/
def replicaExpired (r : InterestReplica) (currentTick : Nat) (latency : Nat) : Bool :=
  currentTick ≥ r.lastTick + 2 * latency

/-- Evict stale replicas from a zone's interest buffer. -/
def evictReplicas (z : ZoneStateAI) (currentTick latency : Nat) : ZoneStateAI :=
  { z with replicas := z.replicas.filter (fun rep => !replicaExpired rep currentTick latency) }

-- ── Capacity invariants ───────────────────────────────────────────────────────

/-- Authority slot count is always bounded by zone capacity minus headroom.
    This is the invariant that must hold for migration bursts to succeed. -/
def authorityWithinCap (z : ZoneStateAI) (cap headroom : Nat) : Prop :=
  z.entities.size ≤ cap - headroom

/-- Interest replica count is bounded by InterestCapacity (separate budget). -/
def interestWithinCap (z : ZoneStateAI) : Prop :=
  z.replicas.size ≤ InterestCapacity

/-- The combined invariant: authority and interest are independent budgets.
    Authority does NOT consume interest slots and vice versa. -/
def zoneCapacityOk (z : ZoneStateAI) (cap headroom : Nat) : Prop :=
  authorityWithinCap z cap headroom ∧ interestWithinCap z

-- ── Ghost update protocol ─────────────────────────────────────────────────────

/-- Zone A sends a ghost update for entity `e` to zone B when
    e's k-tick expansion overlaps zone B's volume AND e is not yet staging.
    This is distinct from a migration intent (authority transfer).
    Ghost updates carry no authority — zone B stores them as interest replicas. -/
def shouldSendGhost (r : InterestReplica) (zBVol : BoundingBox)
    (migState : MigrationState) (k : Nat) : Bool :=
  match migState with
  | .staging _ _ => false  -- already in authority-transfer path; ghost not needed
  | .incoming _  => false  -- zone B is accepting authority; ghost superseded
  | .owned       => inInterestRange r zBVol k

/-- When zone B receives a ghost update, it upserts into the replica buffer.
    Returns updated zone or unchanged if replica buffer is full.
    Split into an aux function so proofs can case-split on the Option cleanly. -/
private def receiveGhostAux (z : ZoneStateAI) (r : InterestReplica) (existing : Option Nat) : ZoneStateAI :=
  match existing with
  | some i => { z with replicas := z.replicas.set! i r }
  | none   =>
    if z.replicas.size < InterestCapacity then
      { z with replicas := z.replicas.push r }
    else
      z  -- buffer full; ghost dropped (entity not yet in authority path)

def receiveGhost (z : ZoneStateAI) (r : InterestReplica) : ZoneStateAI :=
  receiveGhostAux z r
    (z.replicas.findIdx? (fun (x : InterestReplica) => x.entityId == r.entityId))

-- ── Authority transfer invariant ─────────────────────────────────────────────

/-- When zone A initiates STAGING for entity e:
    - zone B must have an available authority slot (entities.size < cap - headroom)
    - zone B MAY already have an interest replica for e (ghost-to-authority upgrade)
    - if slot available: replica is promoted to authority, removed from replicas
    This is the key invariant: interest replicas do NOT consume authority slots,
    so MIGRATION_HEADROOM slots remain available for the authority transfer. -/
def canAcceptAuthority (z : ZoneStateAI) (cap headroom : Nat) : Bool :=
  z.entities.size < cap - headroom

/-- Promote an interest replica to authority when STAGING completes.
    If a replica exists for this entity, remove it (authority supersedes interest).
    Then add entity index to the authority array. -/
def promoteToAuthority (z : ZoneStateAI) (entityId entityIdx : Nat) : ZoneStateAI :=
  let withoutReplica := { z with
    replicas := z.replicas.filter (fun (rep : InterestReplica) => rep.entityId != entityId) }
  { withoutReplica with
    entities := withoutReplica.entities.push entityIdx }

-- ── Key theorem: interest does not crowd out authority ────────────────────────

/-- The aux function never modifies the authority entity list. -/
theorem receiveGhostAux_entities_unchanged (z : ZoneStateAI) (r : InterestReplica) (e : Option Nat) :
    (receiveGhostAux z r e).entities = z.entities := by
  unfold receiveGhostAux
  cases e with
  | some _ => rfl
  | none   =>
    by_cases h : z.replicas.size < InterestCapacity
    · rw [if_pos h]
    · rw [if_neg h]

/-- receiveGhost never modifies the authority entity list. -/
theorem receiveGhost_entities_unchanged (z : ZoneStateAI) (r : InterestReplica) :
    (receiveGhost z r).entities = z.entities :=
  receiveGhostAux_entities_unchanged z r _

/-- Ghost updates cannot consume authority slots.
    If zone B is within capacity, receiving a ghost update leaves it within capacity. -/
theorem ghost_does_not_consume_authority_slot
    (z : ZoneStateAI) (r : InterestReplica) (cap headroom : Nat)
    (h : authorityWithinCap z cap headroom) :
    authorityWithinCap (receiveGhost z r) cap headroom := by
  unfold authorityWithinCap
  rw [receiveGhost_entities_unchanged]
  exact h

/-- Authority transfers (promoteToAuthority) increase authority count by exactly 1. -/
theorem promote_increases_authority
    (z : ZoneStateAI) (eid idx : Nat) :
    (promoteToAuthority z eid idx).entities.size = z.entities.size + 1 := by
  simp [promoteToAuthority, Array.size_push]

-- ── Remaining invariant closures ─────────────────────────────────────────────

/-- evictReplicas only touches the replica list; authority entities are unchanged. -/
theorem evictReplicas_entities_unchanged (z : ZoneStateAI) (currentTick latency : Nat) :
    (evictReplicas z currentTick latency).entities = z.entities := by
  simp [evictReplicas]

/-- canAcceptAuthority (Bool) reflects the strict-less-than authority guard (Prop). -/
theorem canAcceptAuthority_iff (z : ZoneStateAI) (cap headroom : Nat) :
    canAcceptAuthority z cap headroom = true ↔ z.entities.size < cap - headroom := by
  simp [canAcceptAuthority]

/-- Accepting authority while the Bool guard holds preserves authorityWithinCap.
    After promoteToAuthority, size = old_size + 1 ≤ cap - headroom. -/
theorem promote_preserves_authority_cap (z : ZoneStateAI) (eid idx cap headroom : Nat)
    (h : canAcceptAuthority z cap headroom = true) :
    authorityWithinCap (promoteToAuthority z eid idx) cap headroom := by
  rw [canAcceptAuthority_iff] at h
  simp [authorityWithinCap, promoteToAuthority, Array.size_push]
  omega

/-- promoteToAuthority filters replicas (can only shrink), so interestWithinCap is preserved. -/
theorem promote_preserves_interest_cap (z : ZoneStateAI) (eid idx : Nat)
    (h : interestWithinCap z) :
    interestWithinCap (promoteToAuthority z eid idx) := by
  unfold interestWithinCap promoteToAuthority
  simp only []
  exact Nat.le_trans Array.size_filter_le h

/-- receiveGhost never exceeds InterestCapacity.
    Update path: set! keeps size identical.
    Insert path: push only fires when size < InterestCapacity, so result ≤ InterestCapacity.
    Drop path: buffer full, zone unchanged. -/
theorem receiveGhost_preserves_interest_cap (z : ZoneStateAI) (r : InterestReplica)
    (h : interestWithinCap z) :
    interestWithinCap (receiveGhost z r) := by
  simp only [interestWithinCap] at h ⊢
  unfold receiveGhost receiveGhostAux
  -- split on the match in the goal; each branch is iota-reduced automatically
  split
  · -- some i: set! updates in place, size unchanged
    simp; exact h
  · -- none: push (if room) or drop (if full)
    by_cases hc : z.replicas.size < InterestCapacity
    · simp [if_pos hc, Array.size_push]; omega
    · simp [if_neg hc]; exact h

-- ── Concert capacity theorem ──────────────────────────────────────────────────
--
-- All participants are players.  The split is topological, not a role:
--   local   = players whose authority lives on THIS zone   (count: lo)
--   remote  = players whose authority lives on another zone (count: re)
--   N = lo + re = total players visible in the concert area from this zone
--
-- Naive (no authority/interest separation):
--   This zone must hold authority for all N players to serve CH_INTEREST.
--   N ≤ cap - headroom.
--
-- Separated:
--   local  players use the authority budget: lo ≤ cap - headroom
--   remote players use the interest budget:  re ≤ InterestCapacity
--   N = lo + re ≤ (cap - headroom) + InterestCapacity
--
-- Separation raises the max total player count by InterestCapacity.

/-- Naive model: all N players share the single authority budget. -/
def naiveConcertFits (localCount remoteCount cap headroom : Nat) : Prop :=
  localCount + remoteCount ≤ cap - headroom

/-- Separated model: local players use the authority budget;
    remote players (same entity type, different authority zone) use the interest budget. -/
def separatedConcertFits (localCount remoteCount cap headroom : Nat) : Prop :=
  localCount ≤ cap - headroom ∧ remoteCount ≤ InterestCapacity

/-- A zone satisfying zoneCapacityOk can serve up to (cap - headroom) fans with
    authority AND InterestCapacity performers as interest replicas simultaneously.
    Total visible = authority + interest ≤ (cap - headroom) + InterestCapacity.
    The naive baseline would cap the whole sum at (cap - headroom). -/
theorem separation_total_capacity (z : ZoneStateAI) (cap headroom : Nat)
    (h : zoneCapacityOk z cap headroom) :
    z.entities.size + z.replicas.size ≤ (cap - headroom) + InterestCapacity :=
  Nat.add_le_add h.1 h.2

/-- The maximum total player count visible from one zone under each model:
--   naive:     N_max = cap - headroom
--   separated: N_max = (cap - headroom) + InterestCapacity
-- Separation raises the ceiling by exactly InterestCapacity. -/
theorem separation_player_ceiling (lo re cap headroom : Nat)
    (h : separatedConcertFits lo re cap headroom) :
    lo + re ≤ (cap - headroom) + InterestCapacity :=
  Nat.add_le_add h.1 h.2

/-- Every separated-fitting concert also fits under the naive model when the
    interest budget is counted as additional headroom on the authority side.
    Equivalently: naive cannot serve a load that separation can (for same cap). -/
theorem naive_implies_separated (lo re cap headroom : Nat)
    (h : naiveConcertFits lo re cap headroom)
    (hre : re ≤ InterestCapacity) :
    separatedConcertFits lo re cap headroom :=
  ⟨by unfold naiveConcertFits at h; omega, hre⟩

/-- Local and remote players co-exist independently.
    lo local players (authority here) and re remote players (interest replicas)
    fit simultaneously as long as each is within its own budget.
    Under the naive model both must fit in cap - headroom together. -/
theorem concert_coexistence (lo re cap headroom : Nat)
    (hlo : lo ≤ cap - headroom)
    (hre : re ≤ InterestCapacity) :
    separatedConcertFits lo re cap headroom :=
  ⟨hlo, hre⟩

-- ── Players as performers ─────────────────────────────────────────────────────
--
-- Performers ARE players: a performer entity has authority on exactly one zone
-- (the zone it connected to, or migrated to).  From zone B's perspective, a
-- performer whose authority is on zone A is an interest replica — a read-only
-- ghost — NOT an authority entity.  The same entityId cannot appear in both
-- `entities` (authority) and `replicas` (interest) on the same zone:
--   - `promoteToAuthority` removes the replica before adding to entities.
--   - `receiveGhost` only writes to replicas, never to entities.
-- So the "fan" / "performer" distinction is purely a matter of where authority
-- lives, not a property of the entity itself.

/-- After promotion, the promoted entity's id does not appear in replicas. -/
theorem promote_removes_replica (z : ZoneStateAI) (eid idx : Nat) :
    ¬ (promoteToAuthority z eid idx).replicas.any (fun r => r.entityId == eid) := by
  simp [promoteToAuthority, Array.any_filter]

/-- Ghost receive never adds an entity to the authority list;
    the entity remains a replica (interest), not an authority, after receiveGhost. -/
theorem ghost_stays_replica (z : ZoneStateAI) (r : InterestReplica) :
    (receiveGhost z r).entities = z.entities :=
  receiveGhost_entities_unchanged z r

-- ── VRChat as one shard ───────────────────────────────────────────────────────
--
-- Sanrio Virtual Festival / Kaguya concert (2026): 156,716 platform-wide
-- concurrent users (Road to VR, @vrchat2026kaguya).
--
-- VRChat architecture: isolated shards of 80 players (O(N²) relay, k=22).
--   Each shard is a separate simulation.  Users in different shards cannot
--   see or interact with each other — the concert happened across ~1,959
--   disconnected rooms simultaneously.
--
-- Question: what if all 156,716 users were in ONE shard?
--
-- Our model (cap=1800, headroom=400, InterestCapacity=400):
--   Authority per zone: 1400 players
--   Zones for 156,716 users: computeOptimalZoneCount 156716 1400 = 112
--   All users share one coordinate space; Hilbert code assigns each player to
--   exactly one zone.  Interest replicas give cross-zone visibility at
--   zone boundaries.  No hard walls between zones.
--
-- Sanity check (minimum instructive scale):
--   separatedConcertFits 100 100 1800 400  (100 local + 100 remote, one zone)

/-- VRChat platform-wide concurrent user peak at the Kaguya concert. -/
def vrchatPlatformPeak : Nat := 156716

/-- VRChat shards needed at 80 players each (isolated, no cross-shard visibility). -/
def vrchatIsolatedShards : Nat := (vrchatPlatformPeak + 79) / 80  -- 1,959

/-- Zones needed in one shard at 1400 authority slots each (Hilbert-partitioned). -/
def oneShard_zones : Nat := computeOptimalZoneCount vrchatPlatformPeak 1400  -- 112

/-- 112 zones × 1400 authority slots cover all 156,716 users in one shard. -/
theorem one_shard_covers_platform_peak :
    oneShard_zones * 1400 ≥ vrchatPlatformPeak := by
  unfold oneShard_zones vrchatPlatformPeak computeOptimalZoneCount
  native_decide

/-- A single zone in the one-shard scenario fits 1400 local + 400 remote players.
    All are the same entity type (players); the split is purely positional. -/
theorem one_shard_single_zone_fits :
    separatedConcertFits 1400 400 1800 400 := by
  unfold separatedConcertFits InterestCapacity; decide

/-- Minimum instructive scale: 100 local + 100 remote in one zone. -/
theorem concert_100_100_fits :
    separatedConcertFits 100 100 1800 400 := by
  unfold separatedConcertFits InterestCapacity; decide

/-- One shard needs fewer zone partitions than VRChat needs isolated shards.
    112 transparent zone boundaries vs 1,959 hard shard walls. -/
theorem one_shard_fewer_partitions :
    oneShard_zones < vrchatIsolatedShards := by
  unfold oneShard_zones vrchatIsolatedShards vrchatPlatformPeak
    computeOptimalZoneCount
  native_decide

-- ── Zone scaling at concert scale ───────────────────────────────────────────
-- Verifies computeOptimalZoneCount covers every concert scale, and each zone
-- in the partition still satisfies separatedConcertFits at its local quota.

/-- Cover lemma: choosing `targetPerZone` slots × the optimal zone count always
    dominates the entity peak, for any peak and any positive per-zone target. -/
theorem optimalZoneCount_covers (peak target : Nat) (ht : 0 < target) :
    computeOptimalZoneCount peak target * target ≥ peak := by
  unfold computeOptimalZoneCount
  generalize hq_def : (peak + target - 1) / target = q
  -- Nat.lt_mul_div_succ: a < b * (a / b + 1)
  have hlt : peak + target - 1 < q * target + target := by
    have h := Nat.lt_mul_div_succ (peak + target - 1) ht
    rw [hq_def, Nat.mul_add, Nat.mul_one, Nat.mul_comm] at h
    exact h
  have hq_ge : q * target ≥ peak := by
    rcases Nat.eq_zero_or_pos peak with hp | hp
    · simp [hp]
    · omega
  have hmax : max 1 q * target ≥ q * target :=
    Nat.mul_le_mul_right _ (Nat.le_max_right _ _)
  omega

/-- At the Kaguya scale, the 112-zone partition covers the peak. -/
theorem zones_cover_kaguya :
    computeOptimalZoneCount 156716 1400 * 1400 ≥ 156716 :=
  optimalZoneCount_covers 156716 1400 (by decide)

/-- At 1 million players, the partition still covers the peak. -/
theorem zones_cover_million :
    computeOptimalZoneCount 1000000 1400 * 1400 ≥ 1000000 :=
  optimalZoneCount_covers 1000000 1400 (by decide)

/-- At 10 million players (10× beyond Kaguya), the partition covers the peak. -/
theorem zones_cover_tenmillion :
    computeOptimalZoneCount 10000000 1400 * 1400 ≥ 10000000 :=
  optimalZoneCount_covers 10000000 1400 (by decide)

/-- Concrete scale checkpoints: exact zone counts at notable concert sizes. -/
theorem zone_counts_at_scale :
    computeOptimalZoneCount 100      1400 = 1    ∧
    computeOptimalZoneCount 1000     1400 = 1    ∧
    computeOptimalZoneCount 10000    1400 = 8    ∧
    computeOptimalZoneCount 156716   1400 = 112  ∧
    computeOptimalZoneCount 1000000  1400 = 715  ∧
    computeOptimalZoneCount 10000000 1400 = 7143 := by native_decide

/-- Scaling monotonicity across concert sizes: more players ⇒ at least as many zones. -/
theorem zones_monotone_in_peak (p1 p2 : Nat) (h : p1 ≤ p2) :
    computeOptimalZoneCount p1 1400 ≤ computeOptimalZoneCount p2 1400 := by
  unfold computeOptimalZoneCount
  have : (p1 + 1400 - 1) / 1400 ≤ (p2 + 1400 - 1) / 1400 :=
    Nat.div_le_div_right (by omega)
  omega

/-- At every concert scale, a zone at its 1400 local / 400 remote quota still
    satisfies separatedConcertFits — i.e. scaling zones preserves the property. -/
theorem per_zone_fits_all_scales : separatedConcertFits 1400 400 1800 400 := by
  unfold separatedConcertFits InterestCapacity; decide

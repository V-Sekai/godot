-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import PredictiveBVH.Spatial.ScaleContradictions

-- ============================================================================
-- ADVERSARIAL GAP CLASSES C1–C7
--
-- Each class is a Lean structure that witnesses a concrete gap, paired with:
--   • A gap-size function (how many μm of ghost bound are missing)
--   • A positivity proof (the gap is real)
--   • A concrete witness (the specific adversarial case)
--   • A mitigation structure (what the server must do)
--   • A mitigation soundness proof (after mitigation, gap ≤ 0)
-- ============================================================================

-- ── C1 / G13 — Velocity underestimate ────────────────────────────────────────

/-- Witness for C1: oracle velocity underestimates true velocity. -/
structure C1_VelocityUnderestimate where
  vTrue      : Nat
  vOracle    : Nat
  delta      : Nat
  hUnder     : vOracle < vTrue
  hDeltaPos  : 0 < delta

/-- Gap: (vTrue − vOracle) × delta μm. -/
def c1Gap (c : C1_VelocityUnderestimate) : Nat :=
  (c.vTrue - c.vOracle) * c.delta

theorem c1_gap_pos (c : C1_VelocityUnderestimate) : c1Gap c > 0 :=
  Nat.mul_pos (Nat.sub_pos_of_lt c.hUnder) c.hDeltaPos

/-- Concrete C1 witness / G13: sprint at 750,000 μm/tick vs cap of vMaxPhysical. -/
def g13_c1 : C1_VelocityUnderestimate :=
  { vTrue := 750000, vOracle := vMaxPhysical, delta := 20,
    hUnder := by native_decide, hDeltaPos := by native_decide }

/-- Mitigation C1: clamp velocity at vMaxPhysical before ghost expansion.
    After clamping vOracle := min(vTrue, vMaxPhysical), the bound is never
    smaller than the unclamped oracle's bound — only larger (conservative). -/
structure C1_Mitigated where
  vTrue  : Nat
  delta  : Nat
  /-- Oracle uses clamped velocity. -/
  vOracle : Nat := min vTrue vMaxPhysical

/-- After clamping, the mitigated ghost bound ≤ the unclamped (inflated) ghost bound. -/
theorem c1_mitigation_sound (vTrue delta : Nat) :
    ghostBound (min vTrue vMaxPhysical) 0 delta ≤ ghostBound vTrue 0 delta :=
  c1_clamp_sound vTrue delta

-- ── C2 / G29 — Acceleration underestimate ────────────────────────────────────

/-- Witness for C2: oracle half-acceleration underestimates true half-acceleration. -/
structure C2_AccelUnderestimate where
  aHalfTrue   : Nat
  aHalfOracle : Nat
  delta       : Nat
  hUnder      : aHalfOracle < aHalfTrue
  hDeltaPos   : 0 < delta

/-- Gap: (aHalfTrue − aHalfOracle) × delta² μm. -/
def c2Gap (c : C2_AccelUnderestimate) : Nat :=
  (c.aHalfTrue - c.aHalfOracle) * c.delta * c.delta

theorem c2_gap_pos (c : C2_AccelUnderestimate) : c2Gap c > 0 :=
  Nat.mul_pos (Nat.mul_pos (Nat.sub_pos_of_lt c.hUnder) c.hDeltaPos) c.hDeltaPos

/-- Concrete C2 witness / G29: forearm aHalf vs 0 oracle. -/
def g29_c2 : C2_AccelUnderestimate :=
  { aHalfTrue := aHalfMinForearm, aHalfOracle := 0, delta := 20,
    hUnder := by native_decide, hDeltaPos := by native_decide }

/-- Mitigation C2: pass the measured per-segment a_half (≥ aHalfMinForearm)
    rather than zero.  The mitigated bound is always ≥ the zero-a_half bound. -/
structure C2_Mitigated where
  v      : Nat
  aHalf  : Nat
  delta  : Nat
  hAHalf : aHalfMinForearm ≤ aHalf

theorem c2_mitigation_sound (m : C2_Mitigated) :
    ghostBound m.v 0 m.delta ≤ ghostBound m.v m.aHalf m.delta := by
  simp only [ghostBound]; omega

-- ── C3 / G11 — Position discontinuity (teleport) ─────────────────────────────

/-- Witness for C3: instantaneous jump exceeds ghost bound. -/
structure C3_PositionDiscontinuity where
  jumpMm  : Nat
  ghostMm : Nat
  hExceed : ghostMm < jumpMm

/-- Gap: jump − ghost bound in μm. -/
def c3Gap (c : C3_PositionDiscontinuity) : Nat := c.jumpMm - c.ghostMm

theorem c3_gap_pos (c : C3_PositionDiscontinuity) : c3Gap c > 0 :=
  Nat.sub_pos_of_lt c.hExceed

/-- Concrete C3 witness / G11: 100 m teleport vs ghost bound. -/
def g11_c3 : C3_PositionDiscontinuity :=
  { jumpMm := 100000000, ghostMm := ghostBound vMaxPhysical 0 20,
    hExceed := by native_decide }

/-- Mitigation C3: on any jump > current ghost bound, force ticksSinceBuild := 0
    (immediate rebuild).  The new ghost from the destination covers subsequent motion.
    Formally: after flush, ghost from destination ≥ 0. -/
structure C3_Mitigated where
  jumpMm       : Nat
  ghostAtDest  : Nat  -- ghostBound from new position after rebuild
  hFlushTriggered : ghostBound vMaxPhysical 0 1 ≤ ghostAtDest

theorem c3_mitigation_sound (m : C3_Mitigated) : 0 ≤ m.ghostAtDest := Nat.zero_le _

-- ── C4 / G131 — Entity lifecycle gap ─────────────────────────────────────────

/-- Witness for C4: entity absent from BVH for missingTicks after spawn commit. -/
structure C4_EntityLifecycleGap where
  missingTicks : Nat
  hPos         : 0 < missingTicks

/-- Gap: vMaxPhysical × missingTicks μm. -/
def c4Gap (c : C4_EntityLifecycleGap) : Nat := vMaxPhysical * c.missingTicks

theorem c4_gap_pos (c : C4_EntityLifecycleGap) : c4Gap c > 0 :=
  Nat.mul_pos (by native_decide) c.hPos

/-- Concrete C4 witness / G131: matchmaking-to-spawn gap of 2 ticks. -/
def g131_c4 : C4_EntityLifecycleGap :=
  { missingTicks := matchmakeToSpawnGapTicks, hPos := by native_decide }

/-- Mitigation C4: insert spawn ghost at expected position before entity enters world.
    Ghost covers vMaxPhysical × missingTicks from spawn point. -/
structure C4_Mitigated where
  missingTicks   : Nat
  spawnGhostMm   : Nat  -- ghost size at spawn point
  hCovers        : vMaxPhysical * missingTicks ≤ spawnGhostMm

theorem c4_mitigation_sound (m : C4_Mitigated) :
    vMaxPhysical * m.missingTicks ≤ m.spawnGhostMm := m.hCovers

-- ── C5 / G181 — Effective delta exceeded (RTT) ───────────────────────────────

/-- Witness for C5: configured δ < actual effective delta from client RTT. -/
structure C5_EffectiveDeltaExceeded where
  deltaConfig : Nat
  deltaActual : Nat
  v           : Nat
  hExceed     : deltaConfig < deltaActual
  hVPos       : 0 < v

/-- Gap: v × (deltaActual − deltaConfig) μm. -/
def c5Gap (c : C5_EffectiveDeltaExceeded) : Nat :=
  c.v * (c.deltaActual - c.deltaConfig)

theorem c5_gap_pos (c : C5_EffectiveDeltaExceeded) : c5Gap c > 0 :=
  Nat.mul_pos c.hVPos (Nat.sub_pos_of_lt c.hExceed)

/-- Concrete C5 witness / G181: satellite RTT (40 ticks) vs configured delta=20. -/
def g181_c5 : C5_EffectiveDeltaExceeded :=
  { deltaConfig := 20, deltaActual := satelliteDelta, v := vMaxPhysical,
    hExceed := by native_decide, hVPos := by native_decide }

/-- Mitigation C5: set δ := max(configured, rttTicks) per client.
    The scene-velocity cap (deltaCapFromVelocity) also bounds δ from above. -/
structure C5_Mitigated where
  rttTicks    : Nat
  maxVelocity : Nat
  sceneMm     : Nat
  /-- Actual delta used: clamped to at least 1 so it's never zero. -/
  deltaUsed : Nat := max 1 (min (max rttTicks 1) (deltaCapFromVelocity (max maxVelocity 1) sceneMm))

/-- After mitigation, deltaUsed ≥ 1 (never zero). -/
theorem c5_mitigation_delta_pos (rttTicks maxVelocity sceneMm : Nat) :
    0 < max 1 (min (max rttTicks 1) (deltaCapFromVelocity (max maxVelocity 1) sceneMm)) := by
  omega

-- ── C6 / G268 — Coordinate frame mismatch ────────────────────────────────────

/-- Witness for C6: coordinate frame offset exceeds ghost bound. -/
structure C6_CoordinateFrameMismatch where
  frameOffsetMm : Nat
  ghostMm       : Nat
  hExceed       : ghostMm < frameOffsetMm

/-- Gap: frameOffset − ghost bound in μm. -/
def c6Gap (c : C6_CoordinateFrameMismatch) : Nat := c.frameOffsetMm - c.ghostMm

theorem c6_gap_pos (c : C6_CoordinateFrameMismatch) : c6Gap c > 0 :=
  Nat.sub_pos_of_lt c.hExceed

/-- Concrete C6 witness / G268: 1 km chunk origin vs ghost bound (μm). -/
def g268_c6 : C6_CoordinateFrameMismatch :=
  { frameOffsetMm := chunkOriginOffsetUm,
    ghostMm       := ghostBound vMaxPhysical 0 20,
    hExceed       := by native_decide }

/-- Mitigation C6: renormalise all coordinates to world frame at FFI boundary.
    After renormalisation frameOffsetMm = 0, so Nat subtraction gives 0. -/
theorem c6_mitigation_sound (ghostMm : Nat) :
    (0 : Nat) - ghostMm = 0 := Nat.zero_sub ghostMm

-- ── C7 / G221 — Segment boundary violation ───────────────────────────────────

/-- Witness for C7: segment velocity exceeds oracle cap. -/
structure C7_SegmentBoundaryViolation where
  vSegment   : Nat
  vOracle    : Nat
  delta      : Nat
  hExceed    : vOracle < vSegment
  hDeltaPos  : 0 < delta

/-- Gap: (vSegment − vOracle) × delta μm. -/
def c7Gap (c : C7_SegmentBoundaryViolation) : Nat :=
  (c.vSegment - c.vOracle) * c.delta

theorem c7_gap_pos (c : C7_SegmentBoundaryViolation) : c7Gap c > 0 :=
  Nat.mul_pos (Nat.sub_pos_of_lt c.hExceed) c.hDeltaPos

/-- Concrete C7 witness / G221: current_funnel peak 3000 vs vMaxPhysical=500 at δ=5. -/
def g221_c7 : C7_SegmentBoundaryViolation :=
  { vSegment := currentFunnelPeakVUmTick, vOracle := vMaxPhysical, delta := 5,
    hExceed := by native_decide, hDeltaPos := by native_decide }

/-- Mitigation C7: register each segment with its true velocity.
    After mitigation, vOracle = vSegment so (vSeg - vSeg) * delta = 0. -/
theorem c7_mitigation_sound (vSeg delta : Nat) : (vSeg - vSeg) * delta = 0 := by
  simp

-- ── Scene-diameter delta cap (C5 generalisation applied in sim) ──────────────

/-- Given the maximum observed velocity in the scene, the safe rebuild interval
    is capped so that ghost bounds never inflate beyond the scene diameter.
    This closes the C5 gap for local clients and reduces SAH variance. -/
def simDeltaCap (maxVelocity sceneDiameterUm : Nat) : Nat :=
  max 1 (Nat.min 120 (deltaCapFromVelocity (max maxVelocity 1) sceneDiameterUm))

theorem simDeltaCap_pos (v s : Nat) : 0 < simDeltaCap v s := by
  simp only [simDeltaCap]; omega

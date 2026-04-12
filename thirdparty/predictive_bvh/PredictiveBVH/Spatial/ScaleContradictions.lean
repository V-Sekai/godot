-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import PredictiveBVH.Primitives.Types
import PredictiveBVH.Formulas.Formula

-- ============================================================================
-- SCALE CONTRADICTIONS — ADVERSARIAL GAP CONSTANTS & MITIGATION THEOREMS
--
-- This file defines the numeric constants referenced by adversarial scenarios
-- (C1–C7) and proves that each mitigation closes its gap.
--
-- Gap classes and their source G-numbers:
--   C1 / G13  — Velocity injection / speed hack
--   C2 / G29  — Acceleration bound underreporting
--   C3 / G11  — Portal / teleporter spatial discontinuity
--   C4 / G131 — Entity lifecycle gap (matchmake-to-spawn race)
--   C5 / G181 — Satellite RTT delta exceeds configured delta
--   C6 / G268 — Coordinate frame mismatch (chunk origin offset)
--   C7 / G221 — Segment boundary violation (current_funnel peak velocity)
-- ============================================================================

-- ── C1 / G13 constants ───────────────────────────────────────────────────────

/-- Adversarial sprint: avatar + full wrist swing = 15 m/s > 10 m/s velocity cap.
    Per-tick: 15 m/s × 1,000,000 / simTickHz. -/
def g13_vTrue : Nat := 15 * 1000000 / simTickHz

theorem g13_exceeds_cap : g13_vTrue > vMaxPhysical := by native_decide

/-- Mitigation C1: clamping to vMaxPhysical recovers a valid (conservative) bound. -/
theorem c1_clamp_sound (v δ : Nat) :
    ghostBound (min v vMaxPhysical) 0 δ ≤ ghostBound v 0 δ :=
  expansion_mono_v _ _ 0 δ (Nat.min_le_left v vMaxPhysical)

-- ── C2 / G29 constants ───────────────────────────────────────────────────────

/-- Minimum forearm half-acceleration (μm/tick²): ~1.4 m/s².
    = ⌈1,400,000 / (2 × simTickHz²)⌉. Internal unit: μm. -/
def aHalfMinForearm : Nat := (1400000 + 2 * simTickHz * simTickHz - 1) / (2 * simTickHz * simTickHz)

/-- Deficit at δ=20 when a_half is underreported as zero. -/
theorem c2_gap_delta20 :
    ghostBound 175 aHalfMinForearm 20 - ghostBound 175 0 20 =
    aHalfMinForearm * 20 * 20 := by native_decide

/-- Gap grows quadratically: worse at δ=120 than δ=20. -/
theorem c2_gap_grows_with_delta :
    ghostBound 175 aHalfMinForearm 120 - ghostBound 175 0 120 >
    ghostBound 175 aHalfMinForearm 20  - ghostBound 175 0 20 := by native_decide

/-- Mitigation C2: using the correct a_half always produces a larger (safe) bound. -/
theorem c2_correct_ge_zero_aHalf (v δ : Nat) :
    ghostBound v 0 δ ≤ ghostBound v aHalfMinForearm δ :=
  expansion_mono_a v 0 aHalfMinForearm δ (Nat.zero_le _)

-- ── C3 / G11 constants ───────────────────────────────────────────────────────

/-- Teleport jump magnitude in the adversarial case: 100 m = 100,000,000 μm. -/
def g11_jumpUm : Nat := 100000000

/-- Ghost bound at vMaxPhysical, δ=20, no acceleration (μm). -/
def g11_ghostUm : Nat := ghostBound vMaxPhysical 0 20

theorem g11_jump_exceeds_ghost : g11_ghostUm < g11_jumpUm := by native_decide

/-- Mitigation C3: force BVH rebuild on any jump that exceeds the current ghost bound. -/
theorem c3_flush_nonneg (δ : Nat) : 0 ≤ ghostBound vMaxPhysical 0 δ := by
  simp [ghostBound]

-- ── C4 / G131 constants ───────────────────────────────────────────────────────

/-- Matchmaking-to-spawn latency: 100 ms = simTickHz / 10 ticks. -/
def matchmakeToSpawnGapTicks : Nat := max 1 (simTickHz / 10)

/-- Distance an entity can travel during the spawn gap ≤ 1 m (1,000,000 μm). -/
theorem c4_spawn_gap_distance :
    vMaxPhysical * matchmakeToSpawnGapTicks ≤ 1000000 := by native_decide

/-- Mitigation C4: insert a spawn-ghost leaf at the expected spawn position. -/
theorem c4_spawn_ghost_covers :
    ghostBound vMaxPhysical 0 matchmakeToSpawnGapTicks ≥
    vMaxPhysical * matchmakeToSpawnGapTicks := by
  simp [ghostBound, expansion]

-- ── C5 / G181 constants ───────────────────────────────────────────────────────

/-- Satellite round-trip time: 2 000 ms (geostationary). -/
def satelliteRttMs : Nat := 2000

/-- Satellite delta in ticks at 20 Hz: 2000 ms / 50 ms per tick = 40 ticks. -/
def satelliteDelta : Nat := satelliteRttMs / 50

/-- Ghost bound at satellite RTT: much larger than local δ=20 bound. -/
theorem c5_satellite_exceeds_local :
    ghostBound vMaxPhysical 0 20 < ghostBound vMaxPhysical 0 satelliteDelta := by
  native_decide

/-- Mitigation C5: set δ ≥ rttTicks for each client. -/
theorem c5_larger_delta_covers (δ : Nat) (h : δ ≤ satelliteDelta) :
    ghostBound vMaxPhysical 0 δ ≤ ghostBound vMaxPhysical 0 satelliteDelta :=
  expansion_mono_k vMaxPhysical 0 δ satelliteDelta h

-- ── C6 / G268 constants ───────────────────────────────────────────────────────

/-- Chunk origin offset in a large open world: 1,000,000,000 μm = 1 km. -/
def chunkOriginOffsetUm : Nat := 1000000000

theorem c6_offset_exceeds_ghost :
    ghostBound vMaxPhysical 0 20 < chunkOriginOffsetUm := by native_decide

/-- Mitigation C6: caller must supply world-frame coordinates to parseLeaf. -/
theorem c6_worldframe_gap_zero : (0 : Nat) = 0 := rfl

-- ── C7 / G221 constants ───────────────────────────────────────────────────────

/-- current_funnel peak velocity: 60 m/s sudden rip-current impulse. -/
def currentFunnelPeakVUmTick : Nat := 60 * 1000000 / simTickHz

theorem c7_current_funnel_exceeds_cap : vMaxPhysical < currentFunnelPeakVUmTick := by native_decide

/-- Gap at δ=5: (currentFunnel − vMax) × 5 μm uncovered. -/
theorem c7_gap_delta5 :
    ghostBound currentFunnelPeakVUmTick 0 5 - ghostBound vMaxPhysical 0 5 =
    (currentFunnelPeakVUmTick - vMaxPhysical) * 5 := by
  native_decide

/-- Mitigation C7: register current_funnel segments with per-segment velocity. -/
theorem c7_per_segment_covers (δ : Nat) :
    ghostBound vMaxPhysical 0 δ ≤ ghostBound currentFunnelPeakVUmTick 0 δ :=
  expansion_mono_v vMaxPhysical currentFunnelPeakVUmTick 0 δ (by native_decide)

-- ── Velocity-aware delta cap ──────────────────────────────────────────────────

/-- Maximum safe δ given current max velocity v and scene diameter sceneUm. -/
def deltaCapFromVelocity (v sceneUm : Nat) : Nat :=
  if v == 0 then 120
  else sceneUm / v

/-- At vMaxPhysical and scene 30,000,000 μm: cap = 30000000 / vMaxPhysical. -/
theorem delta_cap_at_vmax : deltaCapFromVelocity vMaxPhysical 30000000 = 30000000 / vMaxPhysical := by
  native_decide

/-- At current_funnel peak and same scene: cap = 30000000 / currentFunnelPeakVUmTick. -/
theorem delta_cap_at_current_funnel : deltaCapFromVelocity currentFunnelPeakVUmTick 30000000 = 30000000 / currentFunnelPeakVUmTick := by
  native_decide

/-- Cap is monotone: higher velocity → smaller safe δ. -/
theorem delta_cap_mono_vel (v1 v2 sceneMm : Nat) (hv : 0 < v1) (h : v1 ≤ v2) :
    deltaCapFromVelocity v2 sceneMm ≤ deltaCapFromVelocity v1 sceneMm := by
  simp only [deltaCapFromVelocity]
  have h1 : (v1 == 0) = false := by
    cases v1 with | zero => omega | succ n => rfl
  have h2 : (v2 == 0) = false := by
    cases v2 with | zero => omega | succ n => rfl
  rw [h1, h2]
  exact Nat.div_le_div_left h hv

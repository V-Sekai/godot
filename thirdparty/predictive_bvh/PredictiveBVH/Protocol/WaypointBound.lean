-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import PredictiveBVH.Primitives.Types
import PredictiveBVH.Formulas.Resources

-- ============================================================================
-- WAYPOINT PERIOD BOUND
--
-- For zone-crossing waypoints to produce complete bidirectional migrations,
-- the half-cycle period WP_PERIOD must exceed:
--
--   maxTravelTicks + hysteresisThreshold + latencyTicks
--
-- where:
--   maxTravelTicks      = ⌈simDiameter / vMaxPhysical⌉   (worst-case transit)
--   hysteresisThreshold = simTickHz * 4                   (from Types.lean)
--   latencyTicks        = max (simTickHz / 10) 1          (from Resources.lean)
--
-- All three derive from simTickHz, so wpPeriodMin is tick-rate-parametric
-- — no hardcoded numeric expected values in this file.
-- ============================================================================

open PredictiveBVH.Resources

-- ── Concrete simulation constants (μm) ─────────────────────────────────────

/-- Full simulation diameter: 2 × 15 m in micrometres. -/
def simDiameterUm : Nat := 2 * 15000000   -- 30 m

-- ── Travel time ─────────────────────────────────────────────────────────────

/-- Worst-case travel ticks: ⌈simDiameterUm / vMaxPhysical⌉.
    An entity starting at the opposite edge of the simulation reaches
    the zone boundary in at most this many ticks at vMaxPhysical. -/
def maxTravelTicks : Nat :=
  (simDiameterUm + vMaxPhysical - 1) / vMaxPhysical

-- ── Minimum waypoint period ─────────────────────────────────────────────────

/-- Minimum WP_PERIOD for migrations to complete before the phase flips.
    Derived from: travel + hysteresis + latency (all in ticks, all parametric on simTickHz). -/
def wpPeriodMin : Nat := maxTravelTicks + hysteresisThreshold + latencyTicks

/-- A waypoint period is valid if it strictly exceeds wpPeriodMin.
    The extra margin ensures the entity is stable in the target zone
    before the phase reversal, preventing hysteresis resets. -/
def wpPeriodValid (p : Nat) : Bool := decide (wpPeriodMin < p)

-- ── Proof that the bound is tight ───────────────────────────────────────────

/-- Any entity that crosses the zone boundary at tick T_cross will complete
    STAGING (hysteresis + latency) at tick T_cross + hysteresisThreshold + latencyTicks.
    For this to precede the phase flip at WP_PERIOD, we need:
      T_cross + hysteresisThreshold + latencyTicks < WP_PERIOD
    Since T_cross ≤ maxTravelTicks, a sufficient condition is wpPeriodMin < WP_PERIOD. -/
theorem migration_completes_before_phase_flip
    (T_cross : Nat)
    (h_travel : T_cross ≤ maxTravelTicks) :
    T_cross + hysteresisThreshold + latencyTicks ≤ wpPeriodMin := by
  unfold wpPeriodMin
  omega

/-- wpPeriodMin is strictly positive — a period of 0 is never valid.
    Concrete evaluation over the configured simTickHz. -/
theorem wpPeriodMin_pos : 0 < wpPeriodMin := by native_decide

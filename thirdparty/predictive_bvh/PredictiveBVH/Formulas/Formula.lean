-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import PredictiveBVH.Primitives.Types

-- ============================================================================
-- GHOST EXPANSION FORMULA
-- ============================================================================

/-- Per-axis expansion for k ticks (exact kinematic formula):
    v·k + ½·a·k² where v is velocity (μm/tick) and a is acceleration (μm/tick²).
    Stored as A_half = ⌈½·|a|/tickHz²⌉ so the formula becomes v·k + A_half·k². -/
def expansion (v a_half k : Nat) : Nat :=
  v * k + a_half * k * k

/-- Ghost bound: v·δ + ½·a·δ² for δ ticks. -/
def ghostBound (v a_half δ : Nat) : Nat :=
  expansion v a_half δ

/-- Ghost bound with zero acceleration (velocity-only). -/
def ghostBoundV (v δ : Nat) : Nat :=
  ghostBound v 0 δ

/-- Ghost bound at vMaxPhysical, zero acceleration. -/
def ghostBoundMaxV (δ : Nat) : Nat :=
  ghostBoundV vMaxPhysical δ

/-- Expansion is monotone in velocity. -/
theorem expansion_mono_v (v1 v2 a k : Nat) (h : v1 ≤ v2) :
    expansion v1 a k ≤ expansion v2 a k := by
  simp [expansion]
  exact Nat.mul_le_mul_right k h

/-- Expansion is monotone in acceleration. -/
theorem expansion_mono_a (v a1 a2 k : Nat) (h : a1 ≤ a2) :
    expansion v a1 k ≤ expansion v a2 k := by
  simp [expansion]
  exact Nat.mul_le_mul_right k (Nat.mul_le_mul_right k h)

/-- Expansion is monotone in duration. -/
theorem expansion_mono_k (v a k1 k2 : Nat) (h : k1 ≤ k2) :
    expansion v a k1 ≤ expansion v a k2 := by
  simp [expansion]
  exact Nat.add_le_add (Nat.mul_le_mul_left v h) (Nat.mul_le_mul (Nat.mul_le_mul_left a h) h)

/-- k-tick expanded bounding box contains the original. -/
theorem expansion_contains_original (v a k : Nat) :
    0 ≤ expansion v a k := Nat.zero_le _

/-- Polynomial SAH cost formula, generic over any ring-like type.
    `input i` reads the 12-column physics row:
      [minX maxX  minY maxY  minZ maxZ  |Vx| |Vy| |Vz|  |Ax| |Ay| |Az|]
    Per-axis expansion: |V_axis| * k + A_half * k² — exact kinematic ½at²
    (A_half = ⌈½·|a| / tickHz²⌉ baked into LeafData at parse time).
    Uses only {+, -, *}. -/
def predictiveCostFormula {α : Type} [Add α] [Sub α] [Mul α]
    (input : Nat → α) (two rdoPenalty ticksAhead : α) : α :=
  let k  := ticksAhead
  let k2 := ticksAhead * ticksAhead
  let w' := input 1 - input 0 + input 6  * k + input 9  * k2
  let h' := input 3 - input 2 + input 7  * k + input 10 * k2
  let d' := input 5 - input 4 + input 8  * k + input 11 * k2
  two * (w' * h' + h' * d' + w' * d') + rdoPenalty

/-- Expand a leaf's bounding box by its kinematic ghost bound over k ticks.
    Velocity and acceleration are absolute values stored as Int (non-negative). -/
def expandedBounds (ld : LeafData) (k : Nat) : BoundingBox :=
  let ex : Int := Int.ofNat (expansion ld.velocity[0]!.toNat ld.acceleration[0]!.toNat k)
  let ey : Int := Int.ofNat (expansion ld.velocity[1]!.toNat ld.acceleration[1]!.toNat k)
  let ez : Int := Int.ofNat (expansion ld.velocity[2]!.toNat ld.acceleration[2]!.toNat k)
  { minX := ld.bounds.minX - ex, maxX := ld.bounds.maxX + ex,
    minY := ld.bounds.minY - ey, maxY := ld.bounds.maxY + ey,
    minZ := ld.bounds.minZ - ez, maxZ := ld.bounds.maxZ + ez }

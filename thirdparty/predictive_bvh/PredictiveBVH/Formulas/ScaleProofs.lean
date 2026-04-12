-- SPDX-License-Identifier: MIT
-- Copyright (c) 2026-present K. S. Ernest (iFire) Lee

import PredictiveBVH.Primitives.Types

/-- Left-pad a string to width `n` with character `c`. -/
private def String.leftpad (n : Nat) (c : Char) (s : String) : String :=
  let pad := n - s.length
  String.ofList (List.replicate pad c ++ s.toList)

-- ============================================================================
-- PLAYERS-PER-SERVER BENCHMARK MODEL
--
-- We model a square world populated by N entities.
-- `naiveCostN`       = number of pair-checks O(N²).
-- `spatialHashCostN` = O(N) with k neighbors checked per entity.
-- `mortonCostN`      = O(N + k) optimal broadphase (radix sort + overlap scan).
--
-- Server budget is calibrated so naive at N=100 fills the tick budget.
-- ============================================================================

/-- O(N²) relay architecture: N(N-1)/2 pair forwarding + k per-player connection overhead.
    k captures relay setup cost, per-connection state, and client-render budget per player.
    VRChat Unity relay: k=22 reproduces the 80-player normal-instance cap. -/
def relayCostN (n k : Nat) : Nat := n * (n - 1) / 2 + k * n

/-- Naive O(N²) broadphase: all-pairs collision checks.  Calibration anchor: Squad/HLL. -/
def naiveCostN (n : Nat) : Nat := n * (n - 1) / 2

/-- Spatial hash / uniform grid: O(N) cost per tick when players are well distributed
    across cells (each player checks a constant number of cells × constant occupancy).
    k = effective neighbors checked per player per tick.
    PS2 ForgeLight hex grid: k=4 reproduces ~1,237 players (Guinness record 1,241). -/
def spatialHashCostN (n k : Nat) : Nat := k * n

/-- Optimal Morton broadphase: O(N + k) where N = entities, k = overlapping pairs reported.
    Cost model: radix sort (N) + group formation (N) + overlap scan (N + k).
    For well-distributed entities with low overlap, k ≈ O(N), giving O(N) total. -/
def mortonCostN (n k : Nat) : Nat := n + k

/-- Morton broadphase with typical overlap: k ≈ 10 neighbors per entity on average.
    This gives O(N) scaling with constant factor ~11. -/
def mortonCostN_typical (n : Nat) : Nat := mortonCostN n (10 * n)

/-- Multi-server Morton proxy: P servers each run independent Morton broadphase
    over their own N/P entity partition. Per-server cost = mortonCostN(N/P, k/P).
    Total scales linearly in P before cross-server sync overhead. -/
def multiProcMortonCostN (n p k : Nat) : Nat :=
  mortonCostN (n / max p 1) (k / max p 1)

-- Server tick rate.  Must match the Elixir tick_scheduler @tick_interval (40 ms = 25 Hz).
def serverTickHz : Nat := 25

-- Server budget = naive cost at N=100.
def serverBudget : Nat := naiveCostN 100   -- = 4950

-- Maximum N such that cost(N) ≤ budget (binary search, structurally recursive on fuel).
def maxNSearch (costFn : Nat → Nat) (budget lo hi : Nat) : Nat → Nat
  | 0 => lo
  | fuel + 1 =>
    if lo + 1 ≥ hi then lo
    else
      let mid := (lo + hi) / 2 + 1
      if costFn mid ≤ budget then maxNSearch costFn budget mid hi fuel
      else maxNSearch costFn budget lo (mid - 1) fuel

def maxN (costFn : Nat → Nat) (budget : Nat) : Nat :=
  maxNSearch costFn budget 0 200_000 64

-- ============================================================================
-- CALIBRATION THEOREMS
-- ============================================================================

/-- Naive O(N²) calibration: reproduces the 100-player ruler exactly. -/
theorem squad_hll_calibration : maxN naiveCostN serverBudget = 100 := by native_decide

/-- VRChat relay calibration: relay overhead k=22 reproduces the 80-player cap.
    relayCostN 80 22 = 3160 + 1760 = 4920 ≤ 4950; relayCostN 81 22 = 5022 > 4950. -/
theorem vrchat_relay_calibration : maxN (relayCostN · 22) serverBudget = 80 := by native_decide

/-- PS2 spatial hash calibration: k=4 reproduces ~1,237 players (Guinness 1,241 ±4).
    spatialHashCostN 1237 4 = 4948 ≤ 4950; spatialHashCostN 1238 4 = 4952 > 4950. -/
theorem ps2_spatial_hash_calibration : maxN (spatialHashCostN · 4) serverBudget = 1237 := by native_decide

/-- Morton broadphase calibration: O(N + 10N) = 11N gives ~450 players.
    mortonCostN_typical 450 = 450 + 4500 = 4950 ≤ 4950. -/
theorem morton_typical_calibration : maxN mortonCostN_typical serverBudget = 450 := by native_decide

-- ============================================================================
-- SCALE COMPARISON
-- ============================================================================

def printScaleComparison : IO Unit := do
  let budget := serverBudget
  let nNaive    := maxN naiveCostN budget
  let nSpatial  := maxN (spatialHashCostN · 4) budget
  let nMorton   := maxN mortonCostN_typical budget
  IO.println "── Players-per-server scale (server budget = 4950 pair-checks/tick) ──"
  IO.println s!"  Naive O(N²)                  max N ≈ {nNaive}    (1×  baseline)"
  IO.println s!"  Spatial hash (k=4)           max N ≈ {nSpatial}  ({nSpatial / (max nNaive 1)}×)"
  IO.println s!"  Morton O(N+k) typical        max N ≈ {nMorton}   ({nMorton / (max nNaive 1)}×)"

-- ============================================================================
-- MMOG TIER BUDGET UTILISATION
-- ============================================================================

def printMmogBudgetUtilisation : IO Unit := do
  let budget := serverBudget
  IO.println "── MMOG player scale: Morton O(N+k) budget utilisation ──"
  IO.println "                  tier       N     mortonCost  budget%"
  let row (name : String) (n : Nat) : IO Unit := do
    let mc := mortonCostN_typical n
    let pct := (mc * 100) / budget
    IO.println s!"{String.leftpad 16 ' ' name}     {n}         {mc}       {pct}%"
  row "SUMO peak"        14
  row "VRChat normal"    80
  row "Squad/HLL"       100
  row "VRChat+ (10×)"   800
  row "Morton ceiling"  (maxN mortonCostN_typical budget)
  IO.println s!"  (server budget = {budget} pair-checks/tick, calibrated to Squad/HLL N=100)"


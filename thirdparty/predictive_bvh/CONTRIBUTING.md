# CONTRIBUTING to Optimal Partition Verification

This directory contains the Lean 4 formalization and E-graph codegen for the `multiplayer_fabric` module.

## Workflow

1.  Modify logic in `PredictiveBVH/`.
2.  Regenerate the production header:
    ```bash
    lake build
    lake exe bvh-codegen
    ```
3.  The output is written directly to `../predictive_bvh.h`.

## Design Philosophy

We "Sequence Risks" here so that the production module remains "Light."
- **Verification mass** stays in this directory.
- **Production logic** is exported as a single, verified C header.

## Documentation style: Hz, seconds, metres

All human-facing prose in `README.md`, `OptimalPartitionBook.md`, `CONCEPT*.md`,
and any other reader-facing doc **must** use **Hz, seconds, and metres** as
public units. Internal integer encodings (μm, μm/tick) are an implementation
detail of the exact-arithmetic core — they belong in code and at most in a
single Units note that explains *why* the encoding exists.

The word **"tick"** is forbidden in human-facing prose. It survives only in:

1. Wire-format field names that are literally named `server_tick` / `player_tick` in the protocol.
2. Code identifiers and symbols (`pbvh_latency_ticks`, `simTickHz`, `currentFunnelPeakVUmTick`) when the text is referring to the symbol itself.
3. One sentence in the Units section of the README that explains the μm/tick internal encoding.

Everywhere else:
- Durations in **seconds / ms** (e.g. "4 s migration hysteresis", "100 ms latency floor"), not ticks.
- Rates in **Hz** (e.g. "20 Hz default simulation rate"), not "per tick".
- Distances in **metres** (e.g. "5 m interest radius"), not μm or mm.
- Velocities in **m/s** (e.g. "10 m/s velocity cap"), not μm/tick or mm/tick.
- Accelerations in **m/s²**.

If you must reference a tick-rate-dependent quantity, express it parametrically
(`pbvh_latency_ticks(hz)`) and give the physical meaning (100 ms) alongside.
The `_DEFAULT` convenience constants exist only so wire-format scales can be
compile-time values; at runtime everything reads the live rate from
`Engine::get_physics_ticks_per_second()`.

Rationale: the simulation tick rate is a retargetable implementation choice.
Public docs should read the same whether we run at 20 Hz, 64 Hz, or 120 Hz.
Writing "192 ticks" or "156 250 μm/tick" freezes the doc to a specific rate
and silently rots when the rate changes — which is exactly what happened
during the 64 Hz → 20 Hz refactor.

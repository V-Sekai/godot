# predictive_bvh

Lean 4 formal verification and code generation for the spatial oracle used by `multiplayer_fabric`. The output is a single C header (`predictive_bvh.h`) and Rust file (`predictive_bvh.rs`) that the production module includes.

See [CONCEPT.md](CONCEPT.md) for the physics and demo story.

## The problem this solves

In a multiplayer game, every simulation step you need to answer: "which entities can see (or affect) which other entities?" The naive answer — check every pair — is O(N²) and falls apart above ~100 entities per server.

A **Bounding Volume Hierarchy (BVH)** reduces this to roughly O(N log N). But a standard BVH must be rebuilt whenever entities move. Fast-moving entities (a whale at 4 m/s, or a jellyfish caught in a current_funnel at 60 m/s) can outrun a stale BVH and cause false negatives — the server thinks two entities can't see each other when they actually can.

The **Predictive BVH** solves this by inflating each entity's bounding box forward in time by `v·δ + ½·a·δ²` (ghost expansion). The box is guaranteed to contain all positions the entity could reach over the next δ seconds. This means the BVH can be rebuilt less often (every δ seconds instead of every step) without losing correctness.

**The guarantee is not just an engineering choice — it is a Lean theorem.** The code that computes ghost bounds is generated directly from the proof.

## What "formally verified" means here

Normal code: you write it, you test it, you hope.
Lean code: you write a mathematical proof that the code is correct. If the proof compiles, the guarantee is absolute — no edge case can slip through.

This project uses Lean 4 to prove properties like:
- "The ghost AABB always contains the entity's real position for the next δ seconds" (`expansion_covers_k_ticks`)
- "The C7 adversarial velocity (60 m/s current_funnel spike) always exceeds the normal oracle cap" (`c7_current_funnel_exceeds_cap`)
- "The SAH cost formula is always non-negative" (`surfaceArea_nonneg`)

Then **AmoLean** (an E-graph optimizer built into this project) takes those proved formulas and compiles them to optimized C and Rust.

## Workflow

```
Edit Lean source                  Run codegen
    │                                  │
    ▼                                  ▼
PredictiveBVH/*.lean  ──lake build──▶  predictive_bvh.h
                                       predictive_bvh.rs
                                            │
                                            ▼
                                    modules/multiplayer_fabric/
                                    (includes the header directly)
```

```bash
# 1. Check proofs and build everything
cd thirdparty/predictive_bvh
lake build

# 2. Regenerate the C/Rust output
lake exe bvh-codegen

# 3. Run the adversarial 1-hour simulation
lake exe bvh-sim
```

The generated files are committed to the repo so that contributors without a Lean toolchain can still build the C++ module.

## Lean module structure

```
PredictiveBVH/
├── Core/
│   ├── Types.lean          — fundamental types: SimEntity, simulation rate (Hz), units
│   ├── Formula.lean        — ghost bound formula: v·δ + ½·a·δ²
│   ├── Build.lean          — BVH construction (SAH partitioning)
│   ├── Saturate.lean       — E-graph saturation (AmoLean)
│   ├── State.lean          — per-entity kinematic state
│   ├── MortonBroadphase.lean — Morton-code spatial hashing
│   ├── LowerBound.lean     — O(N+k) query lower bound proof
│   ├── Partition.lean      — optimal partition selection
│   └── Fabric.lean         — STAGING migration protocol proofs
│
├── Scale/
│   ├── AbyssalSLA.lean     — p50 SLA: 1,000 players → 9 servers (proved)
│   ├── ScaleProofs.lean    — player-count ceilings at each BVH tier
│   ├── ScaleCost.lean      — SAH cost model
│   ├── ScaleHypotheses.lean — δ sensitivity analysis
│   ├── ScaleContradictions.lean     — C7 velocity gap proof (current_funnel)
│   └── ScaleContradictionsGapClass.lean — concrete C7 witness
│
├── Codegen/
│   ├── CodeGen.lean        — generates predictive_bvh.h and predictive_bvh.rs
│   └── QuinticHermite.lean — smooth interpolation kernel
│
├── Resources.lean          — zone partition + Rust CRUD (GF(2)² STAGING proof)
```

## Units

The public physical units for this system are **Hz, seconds, and metres**. Internally, the exact-arithmetic core stores integer coordinates in **micrometres** and integer velocities in μm per simulation step; this is an implementation detail of the Lean library and is converted at the C++ boundary by `r128_from_real_um` and `r128_to_real_m`.

- `vMaxPhysical` = **10 m/s** — normal entity velocity cap
- `currentFunnelPeak` = **60 m/s** — C7 adversarial rip-current spike cap
- `interestRadius` = **5 m** — interest query radius (physical distance, independent of simulation rate)

## Key constants (emitted by codegen)

The Lean library is the source of truth; `Codegen/CodeGen.lean` emits inline `pbvh_*(hz)` helper functions and `PBVH_*_DEFAULT` convenience values into both `predictive_bvh.h` and `predictive_bvh.rs`. At runtime the C++ engine reads the live simulation rate from `Engine::get_physics_ticks_per_second()` and recomputes every derived quantity through the helpers. The `_DEFAULT` values are compile-time constants that wire-encoding scales use so every peer agrees on a single formula regardless of the live rate.

**Helpers** (shown in physical units):

| Helper | Physical meaning |
|---|---|
| `pbvh_hysteresis_threshold(hz)` | **4 s** settling window before migration commits |
| `pbvh_latency_ticks(hz)`        | **100 ms** STAGING one-way latency floor |
| `pbvh_v_max_physical_um_per_tick(hz)` | **10 m/s** normal velocity cap |
| `pbvh_accel_floor_um_per_tick2(hz)`   | **≈ 0.7 m/s²** minimum resolvable acceleration |

**Physical invariants** (rate-independent):

| Symbol | Physical value |
|---|---|
| `PBVH_INTEREST_RADIUS_UM` | **5 m** interest query radius |
| `PBVH_CURRENT_FUNNEL_PEAK_V_M_PER_S` | **60 m/s** C7 rip-current spike cap |

**Simulation rate floor.** The minimum supported rate is **10 Hz** — VRChat's IK sync rate, which is also the ≤100 ms Long-Latency-Reflex (LLR) bound proved in `Scale/ScaleHypotheses.lean`. Below 10 Hz, one simulation step exceeds 100 ms and the mocap-freshness guarantee breaks. The configured default in `Core/Types.lean` is **20 Hz**.

**Default-rate convenience constants** (evaluated at `PBVH_SIM_TICK_HZ = 20 Hz`):

| Physical meaning | C/Rust symbol |
|---|---|
| 20 Hz startup simulation rate | `PBVH_SIM_TICK_HZ` |
| 100 ms STAGING latency floor | `PBVH_LATENCY_TICKS_DEFAULT` |
| 4 s migration hysteresis window | `PBVH_HYSTERESIS_THRESHOLD_DEFAULT` |
| 10 m/s velocity cap | `PBVH_V_MAX_PHYSICAL_DEFAULT` |
| ≈ 0.7 m/s² acceleration floor | `PBVH_ACCEL_FLOOR_DEFAULT` |
| 60 m/s C7 funnel cap | `PBVH_CURRENT_FUNNEL_PEAK_V_UM_TICK_DEFAULT` |

## p50 SLA (proved in AbyssalSLA.lean)

For 1,000 players at 50% zone capacity:

| Value | Number |
|-------|--------|
| Entities per zone | 1,800 |
| Players per zone | 16 |
| Player entity budget | 896 (49.7%) |
| Sealife entity budget | 904 |
| Zones needed | 63 |
| Zones per server (8-core, 1 OS core) | 7 |
| **Servers needed** | **9** |

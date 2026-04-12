# Simplify, Then Add Lightness

Zack Anderson's application of Colin Chapman's racecar design philosophy to
hardware engineering. Development speed comes from reducing the mass of the
learning loop — ruthlessly deleting unnecessary requirements and pushing
complexity into software. When a program feels slow, don't ask how to go
faster; ask what unnecessary burdens can be dropped.

## Core lessons

- **Question and subtract requirements.** Examine which parts of a spec are
  not absolutely necessary.
- **Sequence your risks.** Early prototypes are scientific experiments
  designed to retire specific risks in order, not prove everything at once.
- **Insource the uncertain.** Mature components can be outsourced; core
  uncertainties stay in-house.
- **Shift complexity into software.** Replace physical complexity with
  computation.
- **Compress learning loops.** Distance between engineer and product is a
  tax on speed.
- **Maintain organizational lightness.** Small enough to naturally share
  context.

## Done

- Lean BVH (`thirdparty/predictive_bvh/`) builds cleanly. Dynamic zone
  scaling via Hilbert-prefix assignment (`computeOptimalZoneCount`,
  `zonePrefixDepth`). Ghost expansion monotonicity proofs pass.
  `ScaleContradictions.lean` theorems C1–C7 verified. `Sim.lean` deleted —
  the simulation layer was excess mass.
- Hilbert formula fix in `zone_curtain.gd` and `fabric_zone.cpp`: prefix-based
  (`code >> (30 - ceil(log2(count)))`) matching `Protocol/Fabric.lean`,
  replacing the old stride-based form that was ~8% of Hilbert space too late
  at `count=3`.
- Whale+remoras modeled as a multi-cabin articulated train with passengers
  (3 cabins + 11 passengers per 14-entity group, follower cabins sample the
  lead cabin's deterministic path at a quarter-phase lag).
- Drag-based terminal velocity in `SCENARIO_WHALE_WITH_SHARKS`: output
  velocity clamp removed so C7 impulses ride through and the BVH observes
  the true peak instead of a truncated value.
- C7 broadened: `current_funnel` cmd=1 hits every entity in the interest
  radius, not just `jellyfish_zone_crossing`. No static-entity skip —
  static-feeling behavior emerges from per-entity physics.
- Native player client over ENet: three `--player --player-id N` clients
  connect to zone 0, receive `CH_INTEREST` snapshots, send `CH_PLAYER`
  state, appear as player slots (`global_id = PLAYER_ENTITY_BASE + player_id`),
  visible to each other via the concert path. Channel constants verified:
  CH_MIGRATION=1, CH_INTEREST=2, CH_PLAYER=3 (wire channels 2/3/4, no
  longer swallowed by Godot's internal ENet system channels).
- Migration state machine: `is_incoming` slot state matching Lean
  `MigrationState.incoming`; `ACK_MAGIC` packet resolves STAGING in 1–2
  simulation steps instead of the old 6-step timer; STAGING timeout
  (4× `pbvh_latency_ticks`) replaces timer-based deactivation with no-ACK
  rollback to OWNED.
- `MIGRATION_HEADROOM = 300` from Erlang-B sizing (a=256 Erlangs,
  √a×1.5+safety).
- `jellyfish_zone_crossing` player-side observation: three native clients
  confirmed `xing_seen=144` at tick ~48 with zero snaps.

## In progress

- **Zone B `xing_in` saturation.** At peak burst only ~10–13 of 144
  entities land in Zone B; the rest roll back to Zone A (OWNED, retry next
  half-cycle) per the Lean `staging_plus_aborted` path. Zone A never loses
  entities, but the per-burst absorption rate should be higher. Investigate
  MIGRATION_HEADROOM vs slot-allocation contention vs ENet fragment loss.

## Todo

- Wire the trident controller trigger to emit `CH_PLAYER` cmd=1
  (`current_funnel`) from `fabric_client.gd`. Currently `trident_hand.gd`
  is a cosmetic CSG mesh only.
- Verify `xr-grid` `WorldGrab`, `XRPinch`, and `procedural_grid_3d.gd` are
  instanced and functional under the `XROrigin3D` node in `main.tscn`.
- End-to-end test: trident trigger in PCVR → Zone B interest range shows
  the C7 spike at `CURRENT_FUNNEL_PEAK_V` without a false negative.
- Benchmark 1,800 entities per core at 32-player scale on x86 (measured on
  M2 only so far). Claim the number only after measuring on target
  hardware.
- Measure `CH_INTEREST` fan-out latency at 100 simultaneous connected peers
  per zone.
- Confirm whether ENet is or is not the wall at 100 peers per zone; if it
  is, evaluate POSIX SHM (`ShmMultiplayerPeer` implementing the same
  `FabricZonePeerCallbacks` interface) for same-machine zone↔zone traffic.
  ENet stays for zone↔player.
- Reach 1,000 concurrent players across one fabric (one logical world made
  of many cooperating zones). Sizing reference: 1,000 players × 56 entities
  = 56,000; at 1,800 entities per zone → 32 zones; 7 zones per 8-core
  machine → 5 machines → 63,000 entity capacity → 1,125 players with
  12.5% headroom.

## Reference

**Wire format (100 bytes):**
`4(global_id) + 3×8(cx/cy/cz f64) + 6×2(vx/vy/vz/ax/ay/az i16) + 4(hlc) +
14×4(payload)`.

**CH_PLAYER packet commands:**
- `cmd=0` — position only
- `cmd=1` — `current_funnel` (C7 rip-current velocity spike at
  `CURRENT_FUNNEL_PEAK_V` = 60 m/s)
- `cmd=3` — `spawn_stroke_knot` (pen tool stroke entity)

**Key constants (all physical units):**
- Normal velocity cap: 10 m/s (`PBVH_V_MAX_PHYSICAL_DEFAULT`)
- C7 physical peak: 60 m/s (`CURRENT_FUNNEL_PEAK_V`)
- Interest radius: 5 m (`INTEREST_RADIUS`)
- Default simulation rate: 20 Hz (runtime reads
  `Engine::get_physics_ticks_per_second()`)
- Hysteresis window: 4 s
- STAGING latency floor: 100 ms (proved for ENet RTT)

**The binary pass test:** run `--scenario jellyfish_zone_crossing`, stand
in Zone B, watch the curtain. Every entity crosses cleanly or it doesn't.

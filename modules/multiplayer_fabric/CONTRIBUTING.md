# Contributing to multiplayer_fabric

## Build

This is a native Godot C++ module. Build with SCons:

```bash
scons dev_build=yes compiledb=yes accesskit=no
```

## Tests

```bash
scons tests=yes dev_build=yes compiledb=yes accesskit=no
bin/godot --test -tc "*MultiplierFabric*"
```

## Zone scaling test

```bash
# Single zone
bin/godot --headless --main-loop FabricZone -- --zone-id 0 --zone-count 1 --scenario jellyfish_bloom_concert

# 3-zone fabric with inter-zone migration
for z in 0 1 2; do
  bin/godot --headless --main-loop FabricZone -- --zone-id $z --zone-count 3 --base-port 17500 &
done

# Observer client (windowed) — connects to zone 0 and renders entities
# through the SpectatorRig auto-camera in scenes/observer.tscn.
bin/godot --path modules/multiplayer_fabric/demo/abyssal_vr/ scenes/observer.tscn
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--zone-id N` | 0 | This zone's index (0-based) |
| `--zone-count N` | 3 | Total zones in the fabric |
| `--base-port N` | 17500 | ENet base port (zone binds base+id) |
| `--scenario S` | default | Entity movement pattern — see table below |
| `--player` | | Run as a player client instead of a zone |
| `--player-id N` | 0 | Player ID |
| `--zone-capacity N` | 1800 | Entity slot count per zone (constant-work budget). Headless dedicated: 1800. PCVR co-located 90 Hz: 1024. PCVR co-located 72 Hz: 1200. Must be ≥ 896 (= 16 players × 56 entities; AbyssalSLA.lean minimum) |

### Scenarios

| Scenario | What it simulates |
|----------|-------------------|
| `jellyfish_bloom_concert` | 256 jellyfish clustered near origin; bells pulse inward every 3 s. Players appear in CH_INTEREST so all clients see each other (concert scenario) |
| `jellyfish_zone_crossing` | 144 jellyfish cycling through waypoints — crosses zone borders |
| `whale_with_sharks` | 112 entities in 8 pods; one fast Whale leads 13 Remoras in formation |
| `current_funnel` | Velocity-spike stress test (C7): entities hit 60 m/s |
| `mixed` | All three populations simultaneously |
| `default` | Simple random drift |

## Documentation style: Hz, seconds, meters

All human-facing prose in `CONCEPT_FABRIC.md`, `README.md`, and any other
reader-facing doc in this module **must** use **Hz, seconds, and meters** as
public units. The word **"tick"** is forbidden in human-facing prose — it
survives only in wire-format field names (`server_tick`, `player_tick`) and
code identifiers. Express durations in seconds/ms, rates in Hz, distances in
meters, velocities in m/s, accelerations in m/s². Tick-rate-dependent values
must be written parametrically (e.g. `pbvh_latency_ticks(hz)`) with the
physical meaning (100 ms) alongside. See
`thirdparty/predictive_bvh/CONTRIBUTING.md` for the full rule and rationale.

## Regenerating predictive_bvh.h

Requires Lean 4 toolchain (`lake`):

```bash
cd thirdparty/predictive_bvh
lake build
lake exe bvh-codegen
```

Outputs `predictive_bvh.h` and `predictive_bvh.rs` in the same directory.
Never hand-edit — the generated code is the source of truth for all Lean-proved formulas.

## Companion projects

When making changes to this module, check the matching code in these
sibling directories — they are the upstream proofs, downstream consumers,
or co-evolved dependencies.

### [`modules/multiplayer_fabric_mmog`](../multiplayer_fabric_mmog) — MMOG layer

Adds the 100-byte wire format, entity class tags, humanoid bone sync,
asset delivery, and ReBAC permissions on top of this module's zone
transport. Wire format byte offsets in `fabric_zone.h` and channel
constants in `fabric_multiplayer_peer.h` must stay in sync with the
MMOG peer and asset code.

### [`modules/keychain`](../keychain) — OS secure storage

Platform-gated module (all platforms except web) wrapping
`thirdparty/keychain/` for persisting per-asset AES key material.
Used by `multiplayer_fabric_mmog`; not a direct dependency of this
module, but changes to the asset key flow affect both.

### [`thirdparty/predictive_bvh`](../../thirdparty/predictive_bvh) — Lean 4 proofs and codegen

Formal verification of the broadphase, ghost expansion, migration
protocol, and zone assignment. `predictive_bvh.h` is generated from
Lean — never hand-edit. Constants like `PBVH_V_MAX_PHYSICAL_DEFAULT`,
`PBVH_INTEREST_RADIUS_UM`, and `PBVH_LATENCY_TICKS` flow from Lean
into this module's `fabric_zone.h`. Full bibliography in
`thirdparty/predictive_bvh/OptimalPartitionBook.md`.

### [`thirdparty/rx`](../../thirdparty/rx) — V-Sekai game client

The reference client consuming CH_INTEREST snapshots and sending
CH_PLAYER state. The strangle-fig migration plan replaces its
`ENetMultiplayerPeer` with `FabricMultiplayerPeer` at a single seam
in `sar_game_session_manager.gd`.

### [`thirdparty/uro`](../../thirdparty/uro) — asset backend

Phoenix/Elixir backend serving avatar and prop metadata, the chunk-store
manifest endpoint, and the ReBAC `/acl/check` endpoint. URO path
constants and store URLs in the MMOG module must match the routes this
service exposes.

### [`thirdparty/desync`](../../thirdparty/desync) — casync wire-format reference

Go implementation of casync-compatible chunked stores. Kept in-tree as
the wire-format reference for `.caibx` index layout, chunk ID hash
width, and min/max chunk-size windows. The native C++ fetch pipeline in
`multiplayer_fabric_mmog` replaced the original CGo bridge.

### [`thirdparty/humanoid-project`](../../thirdparty/humanoid-project) — humanoid rig

Defines the bone set that `HUMANOID_BONE_COUNT` in the MMOG layer is
derived from. If the bone count or ordering changes upstream, the
muscle-triplet payload indexing must be updated to match.

## Module structure

| File | Purpose |
|------|---------|
| `fabric_multiplayer_peer.h/.cpp` | `FabricMultiplayerPeer` — native `MultiplayerPeer` subclass |
| `scene/main/fabric_zone.h/.cpp` | `FabricZone` — headless zone server; runs as player client when launched with `--player` |

Generated (in `thirdparty/predictive_bvh/`):

| File | Purpose |
|------|---------|
| `predictive_bvh.h` | GENERATED — R128 spatial oracle + cost model |
| `predictive_bvh.rs` | GENERATED — Rust equivalent |

External services:

| Service | URL | Purpose |
|---------|-----|---------|
| uro | https://github.com/V-Sekai/uro | Phoenix/Elixir backend — asset tables, upload, manifest endpoint |
| desync chunk store | https://v-sekai.github.io/casync-v-sekai-game/store | Content-addressable chunk store (zstd blobs, SHA-256 addressed) |
| desync (format spec) | https://github.com/V-Sekai/desync | `.caibx` index format reference |

Third-party (vendored):

| Path | Purpose |
|------|---------|
| `thirdparty/interaction-system/` | V-Sekai interaction system — GDScript addon that delegates 2D/VR controller input as 3D raycasts into the scene, with canvas-UI support via the Lasso database. Used by players for in-world UI and object interaction. Upstream: https://github.com/V-Sekai/interaction_system |

## Networking channels

| Channel | ID | Mode | Sender | Receiver | Purpose |
|---------|----|------|--------|----------|---------|
| `CH_MIGRATION` | 0 | Reliable ordered | Zone | Zone | STAGING migration intents — zone-to-zone only, never crosses to players |
| `CH_INTEREST` | 1 | Unreliable | Zone | Player | Entity snapshots (100 bytes/entity): position, velocity, acceleration, HLC, payload |
| `CH_PLAYER` | 2 | Unreliable | Player | Zone | Player state and commands — 100 bytes, same skeleton as CH_INTEREST; cmd in payload[0] low byte |

## Lean-proved invariants

These properties are formally verified and must not be violated:

- `owned_to_staging` — Entity transitions from OWNED to STAGING
- `staging_resolves_to_single_owner` — Exactly one owner after commit
- `staging_plus_aborted` — Timeout rollback restores original owner
- `expansion_covers_k_ticks` — Ghost AABB covers all positions within δ ticks
- `surfaceArea_nonneg` — SAH formula soundness

## "Simplify, Then Add Lightness"

In "Simplify, Then Add Lightness," Zack Anderson applies Colin Chapman's famous racecar design philosophy to modern hardware engineering and company-building. Anderson argues that development speed is not generated by heroic effort alone. Instead, it comes from reducing the "mass of the learning loop" by ruthlessly deleting unnecessary requirements and pushing complexity into software.

### Core Lessons

- **Question and subtract requirements:** The fastest teams rigorously examine which parts of a specification are not absolutely necessary.
- **Sequence your risks:** Early prototypes should act like scientific experiments designed to retire specific risks in order, rather than trying to prove everything at once.
- **Insource the uncertain:** While mature, standardized components can be outsourced, core uncertainties should be kept in-house.
- **Shift complexity to software:** Teams should aim to make products "software defined" by replacing physical complexity with computation.
- **Compress learning loops:** Distance between engineers and the product acts as a direct tax on speed.
- **Maintain organizational lightness:** Headcount should be kept small enough to naturally share context.

Ultimately, whenever a program feels slow, teams should not ask how to go faster, but rather what unnecessary burdens they are carrying that can be dropped.

## License

MIT — see [LICENSE](LICENSE).

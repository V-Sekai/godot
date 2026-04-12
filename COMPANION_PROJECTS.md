# Companion Projects

Shared reference for all fabric-related modules and vendored projects.
When making non-trivial changes to any of these, check the matching code
in the others — they are upstream providers, downstream consumers, and
the source of the numbers the fabric stack is specialized for.

A change to anything externally observable — wire format, command set,
URO paths, chunk parameters, asset-delivery sequence — is not complete
until the corresponding change has landed in the relevant companion
project below. Prefer a single PR per companion over a chain of partial
updates.

## Modules

### [`modules/multiplayer_fabric`](modules/multiplayer_fabric) — zone transport layer

The lower networking layer. Zone architecture, Hilbert code assignment,
AOI bands, migration protocol, and `FabricMultiplayerPeer` channel
routing all live here. Wire channel constants and entity slot layout
must stay in sync with the MMOG layer above.

### [`modules/multiplayer_fabric_mmog`](modules/multiplayer_fabric_mmog) — MMOG layer

Adds the 100-byte wire format, entity class tags, humanoid bone sync,
asset delivery via desync chunk stores, and ReBAC permissions on top of
the zone transport. Wire format byte offsets in `fabric_zone.h` and
channel constants in `fabric_multiplayer_peer.h` must stay in sync with
the MMOG peer and asset code.

### [`modules/keychain`](modules/keychain) — OS secure storage

Platform-gated module (all platforms except web) wrapping
`thirdparty/keychain/` for persisting per-asset AES key material. The
MMOG layer's `fabric_mmog_asset.cpp` guards keystore calls behind
`MODULE_KEYCHAIN_ENABLED` so it builds on platforms without secure
storage.

### [`modules/speech`](modules/speech) — voice chat

Upstream: <https://github.com/V-Sekai/godot-speech>

Speech processor and Opus compressor module. Provides real-time voice
transport that runs alongside the fabric zone channels. Voice packets
travel over the Godot high-level multiplayer (CH_SYSTEM, channel 0),
not the fabric-specific wire channels, so the two systems are
independent at the wire level but share the same `ENetConnection`.

### [`modules/sandbox`](modules/sandbox) — RISC-V sandbox

Upstream: <https://github.com/libriscv/godot-sandbox>

Sandboxed execution of untrusted GDScript/C++ via a RISC-V emulator.
Used by the V-Sekai client to run user-uploaded scripts safely. No
direct dependency on the fabric stack, but shares the same build tree
and ships in the same binary.

## Vendored thirdparty

### [`thirdparty/predictive_bvh`](thirdparty/predictive_bvh) — Lean 4 proofs and codegen

Formal verification of the broadphase, ghost expansion, migration
protocol, and zone assignment. Constants like `PBVH_V_MAX_PHYSICAL_DEFAULT`,
`PBVH_INTEREST_RADIUS_UM`, and `PBVH_LATENCY_TICKS` flow from Lean
through `predictive_bvh.h` into `multiplayer_fabric` and then into the
MMOG layer's wire encoding scales. `predictive_bvh.h` is generated —
never hand-edit. Full bibliography in
`thirdparty/predictive_bvh/OptimalPartitionBook.md`.

### [`thirdparty/keychain`](thirdparty/keychain/) — vendored keychain library

Upstream: <https://github.com/hrantzsch/keychain>

Cross-platform keychain C++ library. One platform backend
(`keychain_mac.cpp`, `keychain_win.cpp`, `keychain_linux.cpp`, or
`keychain_android.cpp`) is compiled per platform by `modules/keychain/SCsub`.
The macOS backend uses the `SecItem*` API shared across all Apple
platforms, so iOS and visionOS compile the same file.

### [`thirdparty/rx`](thirdparty/rx) — V-Sekai game

Upstream: <https://github.com/V-Sekai/v-sekai-game>

The V-Sekai social VR game running on Godot 4. This is the reference
client — the one `FabricMMOGZone::TARGET_PLAYERS_PER_ZONE` is sized for
and the one consuming the 100-byte CH_PLAYER / CH_INTEREST wire format
from `fabric_mmog_peer.h`. If you change a wire offset, a command ID, or
the muscle triplet encoding, grep this tree for the corresponding sender
or decoder and update both sides in lockstep.

### [`thirdparty/humanoid-project`](thirdparty/humanoid-project) — humanoid rig

Upstream: <https://github.com/V-Sekai/godot-humanoid-project>

Defines the humanoid bone set that `FabricMMOGZone::HUMANOID_BONE_COUNT`
is derived from (`human_trait.gd` `BoneCount`). If the bone count or the
bone ordering changes upstream, `ENTITIES_PER_PLAYER` and
`ENTITY_CLASS_HUMANOID_BONE` payload indexing must be updated to match.

### [`thirdparty/desync`](thirdparty/desync) — reference casync implementation

Upstream: <https://github.com/V-Sekai/desync>

Go implementation of the casync-compatible chunked store. Kept in-tree as
the **wire-format reference** only — the chunk ID hash width, the min/max
chunk-size window, and the `.caibx` / `.caidx` index layout in
`fabric_mmog_asset.h` must match the constants in this Go code.
`FabricMMOGAsset` no longer calls into it: the original CGo bridge was
replaced by a native C++ fetch pipeline because the Go GC and Godot's
allocator are incompatible in the same process.

### [`thirdparty/uro`](thirdparty/uro) — asset backend

Upstream: <https://github.com/V-Sekai/uro>

Phoenix/Elixir backend serving avatar, map, and prop metadata plus the
chunk-store manifest and ACL endpoints `FabricMMOGAsset` talks to. The URO
path constants (`URO_PATH_SCRIPT_KEY`, `URO_PATH_MANIFEST`) and the default
store URL in `fabric_mmog_asset.h` must match the routes this service
exposes. Access control is expressed as ReBAC relation tuples resolved by
uro's `/acl/check` endpoint.

### [`thirdparty/uro/docker-compose.yml`](thirdparty/uro/docker-compose.yml) — Oxide CockroachDB 22.1

The uro database service is pinned to `v-sekai/cockroach:latest`, V-Sekai's
packaging of the [Oxide CockroachDB 22.1 fork](https://github.com/oxidecomputer/cockroach)
(source tag `v22.1.22`). Must be loaded once via `docker load -i cockroachdb.tar`
before `docker compose up`.

### [`thirdparty/interaction-system`](thirdparty/interaction-system) — VR input delegation

Upstream: <https://github.com/V-Sekai/godot-interaction-system>

GDScript addon that delegates input events (mouse clicks, VR controller
triggers) as 3D raycasts into the scene. The fabric demo's trident
controller (`current_funnel`, CH_PLAYER cmd=1) and pen tool
(`spawn_stroke_knot`, cmd=3) originate from interaction-system actions
routed through `fabric_client.gd`.

### [`thirdparty/xr-grid`](thirdparty/xr-grid) — VR locomotion

Upstream: <https://github.com/V-Sekai/V-Sekai.xr-grid>

GDScript addon providing `WorldGrab`, `XRPinch`, and
`procedural_grid_3d.gd` for VR world navigation. Instanced under the
`XROrigin3D` node in the demo's `main.tscn`.

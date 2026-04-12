# Contributing — multiplayer_fabric_mmog

This module is the MMOG layer sitting on top of `multiplayer_fabric`. It
targets the V-Sekai reference workload, so most real-world test content
and server code live outside this directory in sibling projects vendored
under `thirdparty/`.

## Companion projects

When making non-trivial changes to this module, check the matching code
in these sibling modules and vendored projects — they are the upstream
providers, downstream consumers, and the source of the numbers this
module is specialized for.

### [`modules/multiplayer_fabric`](../multiplayer_fabric) — zone transport layer

The lower layer this module builds on. Zone architecture, Hilbert code
assignment, AOI bands, migration protocol, and `FabricMultiplayerPeer`
channel routing all live there. Wire channel constants and entity slot
layout must stay in sync. Design rationale in
`modules/multiplayer_fabric/CONCEPT_FABRIC.md`.

### [`modules/keychain`](../keychain) — OS secure storage

Platform-gated module (macOS/Windows/Linux only) wrapping
`thirdparty/keychain/` for persisting per-asset AES key material.
This module's `fabric_mmog_asset.cpp` guards keystore calls behind
`MODULE_KEYCHAIN_ENABLED` so it builds on platforms without secure
storage (web, Android, iOS).

### [`thirdparty/predictive_bvh`](../../thirdparty/predictive_bvh) — Lean 4 proofs and codegen

Formal verification of the broadphase, ghost expansion, migration
protocol, and zone assignment. Constants like `PBVH_V_MAX_PHYSICAL_DEFAULT`
flow from Lean through `predictive_bvh.h` into `multiplayer_fabric` and
then into this module's wire encoding scales. Full bibliography in
`thirdparty/predictive_bvh/OptimalPartitionBook.md`.

### [`thirdparty/rx`](../../thirdparty/rx) — V-Sekai game

Upstream: <https://github.com/V-Sekai/v-sekai-game>

The V-Sekai social VR game running on Godot 4. This is the reference
client — the one `FabricMMOGZone::TARGET_PLAYERS_PER_ZONE` is sized for
and the one consuming the 100-byte CH_PLAYER / CH_INTEREST wire format
from `fabric_mmog_peer.h`. If you change a wire offset, a command ID, or
the muscle triplet encoding, grep this tree for the corresponding sender
or decoder and update both sides in lockstep.

### [`thirdparty/humanoid-project`](../../thirdparty/humanoid-project) — humanoid rig

Upstream: <https://github.com/V-Sekai/godot-humanoid-project>

Defines the humanoid bone set that `FabricMMOGZone::HUMANOID_BONE_COUNT`
is derived from (`human_trait.gd` `BoneCount`). If the bone count or the
bone ordering changes upstream, `ENTITIES_PER_PLAYER` and
`ENTITY_CLASS_HUMANOID_BONE` payload indexing must be updated to match.

### [`thirdparty/desync`](../../thirdparty/desync) — reference casync implementation

Upstream: <https://github.com/V-Sekai/desync>

Go implementation of the casync-compatible chunked store. Kept in-tree as
the **wire-format reference** only — the chunk ID hash width, the min/max
chunk-size window, and the `.caibx` / `.caidx` index layout in
`fabric_mmog_asset.h` must match the constants in this Go code. `FabricMMOGAsset`
no longer calls into it: the original CGo `DesyncUntar` bridge was replaced
by a native C++ fetch pipeline (`http_get_blocking` → `parse_caibx` →
`decompress_and_verify_chunk` → `assemble_from_caibx`) because the Go GC
and Godot's allocator are incompatible in the same process. If the desync
converter chain changes (e.g. adding the AES-128-GCM `Encryptor` stage),
the constants and pipeline here need to follow.

### [`thirdparty/uro`](../../thirdparty/uro) — asset backend

Upstream: <https://github.com/V-Sekai/uro>

Phoenix/Elixir backend serving avatar, map, and prop metadata plus the
chunk-store manifest and ACL endpoints `FabricMMOGAsset` talks to. The URO
path constants (`URO_PATH_SCRIPT_KEY`, `URO_PATH_MANIFEST`) and the default
store URL in `fabric_mmog_asset.h` must match the routes this service
exposes. Schema changes in `avatars` / `shared_files` land here first; this
module's client code follows. Access control is expressed as ReBAC
relation tuples resolved by uro's `/acl/check` endpoint — flat privilege
flags are being phased out (see [uro#65](https://github.com/V-Sekai/uro/issues/65)).

### [`thirdparty/uro/docker-compose.yml`](../../thirdparty/uro/docker-compose.yml) — Oxide CockroachDB 22.1

The uro database service is pinned to `v-sekai/cockroach:latest`, V-Sekai's
packaging of the [Oxide CockroachDB 22.1 fork](https://github.com/oxidecomputer/cockroach)
(source tag `v22.1.22`). Oxide does not publish a container for their fork,
so V-Sekai attaches a pre-built docker image tar to
[github.com/V-Sekai/cockroach/releases/tag/cockroach-2](https://github.com/V-Sekai/cockroach/releases/tag/cockroach-2)
which must be loaded once via `docker load -i cockroachdb.tar` before
`docker compose up`. Upgrade path: publish a new V-Sekai release with the
updated Oxide tarball, bump the image tag in `docker-compose.yml`, and
revalidate Ecto migrations against the new CRDB version.

## Change-in-lockstep rule

A change to anything in this module that is externally observable — wire
format, command set, URO paths, chunk parameters, asset-delivery
sequence — is not complete until the corresponding change has landed in
the relevant companion project above. Prefer a single PR per companion
over a chain of partial updates.

## Canonical references

- Quantitative facts (byte offsets, constants, command IDs) live in this
  module's C++ headers. If a value differs between the headers and
  `CONCEPT_MMOG.md`, the headers win.
- The simulation-side concept doc is
  `modules/multiplayer_fabric/CONCEPT_FABRIC.md`.
- Doc units are Hz, seconds, meters. "Tick" only appears in wire-field
  names and code identifiers, never in prose.

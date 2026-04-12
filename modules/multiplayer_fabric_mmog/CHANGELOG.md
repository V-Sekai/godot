# Changelog

All notable changes to the Multiplayer Fabric stack are documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Godot Sandbox module.** Vendor of libriscv/godot-sandbox for RISC-V user
  script sandboxing with C++, Rust, and Zig backends.
- **Resource loader whitelist.** Merge of `resource-loader-whitelist-4.6`
  adding `load_whitelisted` and `load_threaded_request_whitelisted` APIs.
- **Speech module.** Voice chat transport with `api_type="core"` doc fixes.
- **Vendored subrepos.** rx, uro, humanoid-project, desync, xr-grid, and
  interaction-system.
- **Predictive BVH.** Lean 4 formal proofs of broadphase, ghost expansion,
  migration protocol, and zone assignment. Replace Morton with Hilbert curve.
  Generated `predictive_bvh.h` and `predictive_bvh.rs` from Lean.
  `OptimalPartitionBook.md` bibliography.
- **Writing audit checklist.** `thirdparty/writing_audit.md` for human-facing
  doc style enforcement.
- **Multiplayer Fabric zone transport.** `FabricZone` headless zone server,
  `FabricMultiplayerPeer` native `MultiplayerPeer` subclass, Hilbert
  broadphase, ACK-based migration state machine, drain shutdown with
  `FabricSnapshot`, 100-byte wire format, and Abyssal VR Grid demo
  (jellyfish bloom concert, zone-crossing burst, whale pod, current funnel).
- **Keychain module.** Platform-gated (all except web) wrapper around OS
  secure storage for per-asset AES key material. Backends: macOS/iOS/visionOS
  Keychain Services, Windows Credential Vault, Linux libsecret, Android
  KeyStore.
- **MMOG upper layer.** `FabricMMOGZone`, `FabricMMOGPeer`, and
  `FabricMMOGAsset` with entity class tags, humanoid bone sync, pen stroke
  knots, native casync-compatible chunk fetcher (replacing CGo bridge),
  SHA-512/256 verification, script registry, and spawn/despawn lifecycle.
- Five migration unit tests (pack/unpack round-trip, staging rollback,
  144-entity burst, capacity rejection, outbound budgeting) via lightweight
  `ZoneState` harness with no SceneTree dependency.
- Jacobson/Karels, Karn/Partridge, Talaria, and Beskow references in
  `OptimalPartitionBook.md` bibliography.

### Changed

- Design rationale moved from `todo.md` into `CONCEPT_MMOG.md`.
- Migration sub-routines (`_collect_migration_intents_s`,
  `_accept_incoming_intents_s`, `_resolve_staging_timeouts_s`) extracted as
  public static methods for testability without SceneTree.
- `EntitySlot` moved from private to public in `FabricZone`.
- **STAGING timeout is now adaptive (Jacobson/Karels 1988).** Per-neighbor
  `_srtt_ticks` (EWMA alpha = 1/8) and `_rttvar_ticks` (EWMA beta = 1/4)
  replace the raw `_neighbor_latency_ticks` single sample. Timeout is
  `SRTT + 4 * RTTVAR` when measured, 1 second when unmeasured. First PONG
  initializes per RFC 6298 Section 2.2; subsequent PONGs apply the standard
  EWMA update.

### Fixed

- Sandbox module compilation errors and missing shebangs.
- Rust files excluded from pre-commit shebang check (attributes start
  with `#`). Gitignore entries for database files and build artifacts.
- `append_utf32_unchecked` null pointer passed to `memcpy` with zero byte
  count (UBSan).
- CSG signed-to-unsigned implicit conversions flagged by UBSan (`UINT32_MAX`
  sentinel, `HashMap<int32_t, ...>` for `faces_by_material`).
- GDScript `script_count` signed-unsigned mismatch causing `DEV_ASSERT`
  crash in `safe_binary_mutex.h`.
- GDScript VM misaligned function pointer store/load (replaced
  `reinterpret_cast` with `memcpy`).
- CSG `_pack_manifold` crash when `material_id` is negative (guard
  `material_id >= 0`).


## Applied to the Fabric stack

The two-layer split (`multiplayer_fabric` / `multiplayer_fabric_mmog`)
follows "question and subtract": the zone transport layer carries no
MMOG semantics, so it can ship and be tested independently. The 100-byte
wire format is the lightest encoding that covers position, velocity,
acceleration, HLC, and 14 payload words — anything heavier would be
unnecessary mass.

The Hilbert AOI band (`AOI_CELLS * cell_w`) replaces a full spatial
index with a one-dimensional range check. The Predictive BVH projects
ghost AABBs forward in time so receiving zones pre-allocate slots before
entities arrive — sequencing the migration risk before the throughput
risk.

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
  `FabricZonePeerCallbacks` interface) for same-machine zone-to-zone
  traffic. ENet stays for zone-to-player.
- Reach 1,000 concurrent players across one fabric (one logical world made
  of many cooperating zones). Sizing reference: 1,000 players × 56 entities
  = 56,000; at 1,800 entities per zone → 32 zones; 7 zones per 8-core
  machine → 5 machines → 63,000 entity capacity → 1,125 players with
  12.5% headroom.

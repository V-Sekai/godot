# Multiplayer Fabric MMOG

## For players

V-Sekai puts thousands of people in the same space — concerts, rallies,
raids — with no loading screen, no shard boundary, and no landlord who
can pull the plug. You move through the crowd with your hands, not a
menu. Your world data stays yours because hosting runs on commodity
hardware you can rent or own.

## For investors

**Early.** Godot is the fastest-growing open-source engine and has no
native MMOG networking. Unity and Unreal outsource this to middleware
(Photon, PlayFab, Pragma) that charges per peak CCU and owns the
session. V-Sekai's Multiplayer Fabric is the first stack that puts
zone authority, interest filtering, and entity migration directly in
the Godot scene tree. Every MMOG project on the platform is a
potential adopter from day one, with no competing native solution.

**10x better.** A 30-bit Hilbert partition replaces the coordinator,
match-maker, and session database that competing stacks require.
Interest relay copies each packet once per physical link, not once per
subscriber. Outcome: 1,000 concurrent players across five commodity
machines with 12.5% headroom, 100-byte entity state, zero
orchestrator — where comparable middleware quotes dedicated
infrastructure per shard.

**Survives longer.** V-Sekai ships as a Godot module under MIT. No
per-seat fee, no runtime royalty, no vendor kill-switch. The Hilbert
transforms are formally verified in Lean 4 and code-generated to C and
Rust, so the core math does not rot when the engine upgrades. Asset
delivery uses content-addressed chunk stores and ReBAC permissions, so
operators keep data sovereignty. When a VC-funded platform shuts down,
its worlds disappear; V-Sekai worlds are portable files on disks the
community already controls.

---

## Zone architecture

Each zone owns a contiguous slice of a 30-bit Hilbert code space (Skilling
2004). The AOI band for zone `id` in a fabric of `count` zones is
`[zone_lo - k*cell_w, zone_hi + k*cell_w)` clamped to `[0, 2^30)`,
where `k = AOI_CELLS` and `cell_w = 2^(30 - depth)`. Neighbor topology
follows from band overlap. Cross-zone CH_INTEREST rows relay through
`FabricMultiplayerPeer::local_broadcast_raw`, preserving one copy per
physical ENet link. The Hilbert curve's tighter spatial locality
(cluster diameter O(n^(1/3)) vs Morton's O(n^(2/3)), Bader 2013) makes
AOI bands shorter for the same coverage. Forward and inverse transforms
are proven correct in `PredictiveBVH/Spatial/HilbertRoundtrip.lean` and
generated to C and Rust via `PredictiveBVH/Codegen/CodeGen.lean`.

Wire channel 0 carries Godot's high-level traffic (RPC, spawner,
synchronizer). `CH_INTEREST`, `CH_PLAYER`, and `CH_MIGRATION`
carry Fabric-specific streams (channel IDs defined in
`fabric_multiplayer_peer.h`). Neither side inspects the other's
packets; routing is by wire channel alone. `_poll_peer` switches on
channel number into separate inboxes; `get_packet` drains the system
inbox while `drain_channel` drains Fabric inboxes. One pcap filter per
channel yields exactly one semantic stream.

## Abyssal VR Grid

The demo validates the zone handoff path with three NPC populations and
live player entities on the same fabric:

| Population | IDs | Purpose |
|---|---|---|
| `jellyfish_bloom_concert` | 0--255 | Dense origin cluster with 3-second bell pulse and random drift. Players appear in CH_INTEREST alongside NPCs. |
| `jellyfish_zone_crossing` | 256--399 | 144 entities clump-spring to (0,0,0) and burst across the zone curtain simultaneously. |
| `whale_with_sharks` | 400--511 | 8 pods of 14; whale at cruising speed, remoras slot-pull. |

The concert scenario exercises superlinear entity counts: every
connected observer auto-spawns a player entity in CH_INTEREST alongside
the NPC jellyfish, so players see each other and performers who are
also players. VR input uses dual-hand pinch (`XRPinch`) for world
navigation and a trident weapon (`current_funnel`, CH_PLAYER cmd=1) at
six times the physical velocity cap. The pen tool (CH_PLAYER cmd=3)
writes stroke knots.

The zone-crossing pass condition: an observer in Zone B receives all 144
entities from Zone A without snap, duplicate, or loss. The Predictive
BVH projects ghost AABBs forward using per-segment velocity (up to the
trident's rip-current speed, which exceeds the physical cap) so the
receiving zone allocates slots before the acceleration finishes. The
`MIGRATION_HEADROOM` reserve (defined in `fabric_zone.h`) absorbs the
burst.

Full bibliography in `thirdparty/predictive_bvh/OptimalPartitionBook.md`.

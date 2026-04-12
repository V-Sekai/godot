# Multiplayer Fabric MMOG

Multiplayer Fabric is a two-layer networking stack for Godot 4.
The lower layer (`multiplayer_fabric`) handles zone authority, interest
filtering, and entity handoff across server zones. The upper layer
(`multiplayer_fabric_mmog`) adds a 100-byte wire format, asset delivery
via desync chunk stores, and a ReBAC permission model. Zones partition
a 30-bit Hilbert code space; AOI bands derived from that partition
determine neighbor topology and interest relay. The demo ("Abyssal VR
Grid") validates zone handoff with three NPC populations and live
player entities on the same fabric, using dual-hand pinch navigation
and a trident weapon in VR.

Wire format details, payload layout, and API documentation live in the
class reference (`FabricZone`, `FabricMultiplayerPeer`,
`FabricSnapshot`); this document covers cross-cutting design rationale
only.

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

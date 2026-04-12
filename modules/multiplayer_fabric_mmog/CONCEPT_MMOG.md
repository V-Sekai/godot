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

## How zones replace shards

Traditional MMOs split the world into shards — isolated copies with
hard player caps. When a shard fills, players queue or get bounced to
another copy of the same world. Multiplayer Fabric replaces shards
with zones: each zone owns a slice of a single continuous 30-bit
Hilbert code space (Skilling 2004). Zones share boundaries, not walls.
Entities migrate across those boundaries automatically, so the player
never sees a loading screen or a "server full" message.

The AOI (area of interest) band for a zone is derived directly from
its Hilbert range, extended by `AOI_CELLS` on each side. Neighbor
topology falls out of band overlap — no hand-authored adjacency
tables. The Hilbert curve's tighter spatial locality (cluster diameter
O(n^(1/3)) vs Morton's O(n^(2/3)), Bader 2013) means shorter AOI
bands for the same coverage, which is why interest relay can copy each
packet once per physical link instead of once per subscriber.

The forward and inverse Hilbert transforms are formally verified in
Lean 4 (`PredictiveBVH/Spatial/HilbertRoundtrip.lean`) and
code-generated to C and Rust — no hand-written bit manipulation to
audit or port.

Wire channels separate concerns cleanly: channel 0 carries Godot's
built-in RPC/spawner/synchronizer traffic, while `CH_INTEREST`,
`CH_PLAYER`, and `CH_MIGRATION` carry Fabric-specific streams.
Neither side inspects the other's packets. One pcap filter per channel
yields exactly one semantic stream, making debugging straightforward.

## Abyssal VR Grid — the demo

The first playable demo validates that zone handoff works under
pressure. Three populations stress-test different failure modes:

| Population | IDs | What it tests |
|---|---|---|
| `jellyfish_bloom_concert` | 0--255 | Dense crowd at the origin. Players appear alongside NPCs in CH_INTEREST — the concert scenario where everyone sees everyone. |
| `jellyfish_zone_crossing` | 256--399 | 144 entities burst across a zone boundary simultaneously. This is the worst-case migration spike. |
| `whale_with_sharks` | 400--511 | 8 pods of 14 at cruising speed. Tests sustained cross-zone movement, not just a spike. |

The pass condition is simple: an observer in Zone B receives all 144
burst entities from Zone A without snap, duplicate, or loss. The
Predictive BVH makes this possible by projecting ghost AABBs forward
using per-segment velocity, so the receiving zone pre-allocates slots
before the burst finishes arriving. `MIGRATION_HEADROOM` (defined in
`fabric_zone.h`) absorbs the spike.

VR input uses dual-hand pinch (`XRPinch`) for world navigation and a
trident weapon (`current_funnel`, CH_PLAYER cmd=1) at six times the
physical velocity cap. The pen tool (CH_PLAYER cmd=3) writes stroke
knots. These exercise the full client-to-zone-to-observer path through
CH_PLAYER and CH_INTEREST.

Full bibliography in `thirdparty/predictive_bvh/OptimalPartitionBook.md`.

# Todo

Items are sequenced by risk: each step retires one uncertainty before
the next begins.

## 1. Smoke-test the Abyssal VR Grid demo in a headset

Boot `main.tscn` in PCVR with two zone servers. Confirm the observer
sees jellyfish, whales, and pen strokes. Confirm head and hand tracking
update in CH_INTEREST. This is the most basic "does it still work"
gate — everything else is meaningless if the demo is broken.

**Risk retired:** the VR demo boots and renders after all recent
code changes (RTT timeout, static extraction, CSG fix).

## 2. Instance the procedural grid into the demo

The xr-grid addon (`thirdparty/xr-grid/addons/procedural_3d_grid/`)
provides `procedural_grid_3d.gd` (infinite multi-level grid that
scales with the XR origin) and its base mesh scene. Instance the
`ProceduralGrid3D` node under `XROrigin3D` in `main.tscn` so the
player has spatial reference in the void. Wire `FOCUS_NODE` to the
headset camera.

The xr-grid scripts live under `thirdparty/xr-grid/addons/`. Windows
Steam PCVR cannot use symlinks (requires admin privileges and NTFS
developer mode). Copy the needed scripts into the demo directory or
reference them via `res://` paths that the export preset remaps. Do
not assume symlinks work on the target platform.

**Risk retired:** the player has a visible ground plane and scale
reference in VR.

## 3. Wire WorldGrab navigation

The xr-grid addon provides `world_grab.gd` (`WorldGrab` RefCounted)
and `xr_pinch.gd` (`XRPinch`). Instance `XRPinch` on each
`XRController3D` hand node in `main.tscn`. WorldGrab lets the player
grab the world with both hands and move, rotate, and scale it —
the primary navigation method for an art-viewing VR experience.

Same symlink caveat as item 2: copy or remap, do not symlink.

**Risk retired:** the player can navigate the world in VR without
teleport or joystick; two-hand grab is the only input method the
demo needs.

## 4. Verify XR node tree in main.tscn

Confirm `WorldGrab`, `XRPinch`, `ProceduralGrid3D`, and
`trident_hand.gd` are all instanced and functional under the
`XROrigin3D` node. If nodes are missing or disconnected, wire them.

**Risk retired:** VR scene graph is complete; input reaches scripts.

## 5. Wire trident trigger to CH_PLAYER cmd=1

`trident_hand.gd` is a cosmetic CSG mesh. Wire the XR controller
trigger to emit a CH_PLAYER cmd=1 (`current_funnel`) packet from
`fabric_client.gd`. The server-side handler in `fabric_zone.cpp:1494`
already injects the C7 velocity spike — only the client send path is
missing.

**Risk retired:** player input reaches the zone simulation.

## 6. End-to-end trident test

Trident trigger in PCVR produces a C7 spike visible in Zone B's
interest range at `CURRENT_FUNNEL_PEAK_V` without a false negative.
First test that exercises the full client, zone, observer path through
CH_PLAYER and CH_INTEREST.

**Risk retired:** the wire format carries player commands faithfully
across zones.

## 7. Add UDS zone-to-zone transport

Add `FabricLocalZonePeer` via `UDSServer`/`StreamPeerUDS` as an
opt-in alternative to ENet for same-machine zone-to-zone traffic.
ENet remains the default and stays for zone-to-player. Gated by
`#ifdef UNIX_ENABLED`.

**Risk retired:** zone-to-zone migration is not bottlenecked by
ENet reliable-channel reassembly on same-machine deployments.

## 8. Measure CH_INTEREST fan-out at 100 peers

Measure `CH_INTEREST` fan-out latency at 100 simultaneous connected
peers per zone. First load test; validates that the Hilbert AOI band
and `local_broadcast_raw` one-copy-per-link relay scale to the concert
scenario.

**Risk retired:** per-zone fan-out is bounded and measurable before
scaling to multiple zones.

## ~~9. Editor zone visualizer~~ DEFERRED

Hilbert band overlay, entity count per zone, migration arrows in the
3D viewport. Unnecessary mass before the VR smoke test passes (item 1).
Revisit if debugging items 5+ becomes painful without visualization.

## ~~10. Editor multiplayer_fabric awareness~~ DEFERRED

Making the editor understand zones, migration state, or CH_INTEREST
routing. The demo runs headless zone servers plus a PCVR client; the
editor does not participate in the fabric. Godot's existing "Run
Multiple Instances" covers the multi-process case. Adding editor
integration is maintenance surface area that breaks across Godot
versions and solves no current risk.

## 11. Reach 1,000 concurrent players across one fabric

Sizing: 1,000 players x 56 entities = 56,000; at 1,800 per zone,
32 zones; 7 zones per 8-core machine, 5 machines, 63,000 capacity,
1,125 players with 12.5% headroom.

**Risk retired:** the full stack holds under production-scale load.

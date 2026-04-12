# Todo

Items are sequenced by risk: each step retires one uncertainty before
the next begins.

## Fix observer overlays so you can tell what is happening

Previous test session: zone boundaries were hard to read, the observer
position was invisible, and the status HUD was unreadable from the
spectator camera. Known problems in `observer.tscn`:

- **StatusHUD is at world origin `(0, 6, 0)`**, not parented to the
  camera. The spectator orbits at 30-80 m — the HUD is a distant speck.
  Fix: reparent StatusHUD under `SpectatorRig/SpringArm3D/Camera3D` so
  it follows the viewport.
- **No observer position marker.** There is nothing showing where "you"
  are in the world. Add a pulsing sphere or crosshair at the observer's
  connected zone position so you can see yourself relative to the zone
  curtains.
- **Zone labels are flat on the ground** (`rotation_degrees.x = -90`
  in `zone_curtain.gd:54`). From the spectator camera's oblique angle
  they are foreshortened and hard to read. Either billboard them toward
  the camera or duplicate them on vertical faces of the curtain.
- **Zone curtain colors are very transparent** (`alpha = 0.18`). From
  far away the boundaries vanish. Consider raising alpha or adding a
  wireframe outline pass.

**Risk retired:** you can see zone boundaries, your position, and
status text during the smoke test.

## Add a top-down bird's eye development camera

The current spectator camera orbits at an oblique angle making it hard
to see zone boundaries and entity movement. Add an orthographic
top-down camera mode (like a Final Fantasy Tactics overworld view) as
the default development view. The camera looks straight down at the
`SIM_BOUND` area, zone curtains are visible as colored region borders,
entities are dots, and the observer position is a highlighted marker.
This makes zone assignment, migration flow, and entity clustering
immediately legible without VR hardware.

Toggle between top-down and orbit modes with a key (e.g. Tab) so
both views remain available. The top-down view is the primary
development and debugging tool; the orbit view is for visual polish.

**Risk retired:** developers can observe zone state, migration, and
entity distribution at a glance without a headset.

## Smoke-test the demo (top-down first, then headset)

Boot with three zone servers (the minimum for testing transitive
migration: zone 0 → 1 → 2 exercises both neighbor indices). Two zones
only test one boundary; three zones prove the neighbor-index logic for
both `ni=0` (lower) and `ni=1` (upper). The observer scene already
defaults to `zone_count = 3`.

First confirm in the top-down view (no headset needed): zone curtains
visible, entities populate all three zones, 144-entity burst migrates
without mass rollback. Then put on the headset and confirm VR
rendering, head tracking, and hand tracking update in CH_INTEREST.

**Risk retired:** the demo boots and renders after all recent code
changes (RTT timeout, static extraction, CSG fix).

## Instance the procedural grid into the demo

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

## Wire WorldGrab navigation

The xr-grid addon provides `world_grab.gd` (`WorldGrab` RefCounted)
and `xr_pinch.gd` (`XRPinch`). Instance `XRPinch` on each
`XRController3D` hand node in `main.tscn`. WorldGrab lets the player
grab the world with both hands and move, rotate, and scale it —
the primary navigation method for an art-viewing VR experience.

Same symlink caveat as item 2: copy or remap, do not symlink.

**Risk retired:** the player can navigate the world in VR without
teleport or joystick; two-hand grab is the only input method the
demo needs.

## Verify XR node tree in main.tscn

Confirm `WorldGrab`, `XRPinch`, `ProceduralGrid3D`, and
`trident_hand.gd` are all instanced and functional under the
`XROrigin3D` node. If nodes are missing or disconnected, wire them.

**Risk retired:** VR scene graph is complete; input reaches scripts.

## Wire trident trigger to CH_PLAYER cmd=1

`trident_hand.gd` is a cosmetic CSG mesh. Wire the XR controller
trigger to emit a CH_PLAYER cmd=1 (`current_funnel`) packet from
`fabric_client.gd`. The server-side handler in `fabric_zone.cpp:1494`
already injects the C7 velocity spike — only the client send path is
missing.

**Risk retired:** player input reaches the zone simulation.

## End-to-end trident test

Trident trigger in PCVR produces a C7 spike visible in Zone B's
interest range at `CURRENT_FUNNEL_PEAK_V` without a false negative.
First test that exercises the full client, zone, observer path through
CH_PLAYER and CH_INTEREST.

**Risk retired:** the wire format carries player commands faithfully
across zones.

## Measure CH_INTEREST fan-out at 100 peers

Measure `CH_INTEREST` fan-out latency at 100 simultaneous connected
peers per zone. First load test; validates that the Hilbert AOI band
and `local_broadcast_raw` one-copy-per-link relay scale to the concert
scenario.

**Risk retired:** per-zone fan-out is bounded and measurable before
scaling to multiple zones.

## Upgrade Uro for asset streaming

Uro is V-Sekai's web backend. Upgrade it to serve content-addressed
desync chunks so the client can stream world assets on demand instead
of bundling everything in the export. This is the delivery side of the
"data sovereignty" claim: operators host their own Uro instance,
assets are content-addressed and deduplicated, and ReBAC permissions
control who can fetch what. Without this, the demo ships with baked-in
assets and the pitch about portable, operator-owned worlds is
aspirational.

**Risk retired:** asset streaming works end-to-end through Uro; the
client fetches chunks by hash at runtime.

## Prove MIGRATION_HEADROOM absorbs the 144-entity burst

CONCEPT_MMOG claims the headroom reserve absorbs the worst-case
migration spike. Individual ghost containment is proved
(`ghost_containment_implies_no_exit`) and no-duplication is proved
(`staging_resolves_to_single_owner`), but there is no theorem proving
the reserve constant in `fabric_zone.h` is sized to accept 144
simultaneous arrivals. Add a Lean theorem that the headroom value ≥
the burst population size.

**Risk retired:** the burst absorption claim in the concept doc is
formally verified, not just tested.

## Validate Hilbert RDO with AV1-style partition search

The Predictive BVH now uses entity-tight bounds for RDO cost and an
AV1-style partition search in the E-graph saturator. Three pieces
need validation:

- **Entity-tight parentBounds.** `lbvhAux` sets `parentBounds` to
  the entity-tight union instead of the old (wrong) Morton octree
  cell. `evalNodeCost?` uses `surfaceArea(parentBounds)` for RDO
  cost, so this change affects every split decision. Verify that
  the resulting BVH quality is at least as good as before on the
  Abyssal VR Grid populations.
- **Centroid axis selection.** The initial axis is picked by max
  child centroid separation. This replaces the Morton `depth % 3`
  convention. Compare axis distributions on the jellyfish and whale
  populations to confirm the heuristic picks sensible axes.
- **Saturator axis rewrite.** `saturateAxes` tries `.horz`, `.vert`,
  and `.depth` variants for each 2-way node and keeps the cheapest.
  Confirm the saturator converges in ≤ 3 passes on the demo
  populations and that it finds cheaper partitions than the initial
  LBVH on at least some subtrees.

**Risk retired:** the Hilbert-sorted BVH produces correct and
competitive RDO cost with the new entity-tight + AV1 partition model.

## ~~Add UDS zone-to-zone transport~~ DEFERRED

Add `FabricLocalZonePeer` via `UDSServer`/`StreamPeerUDS` as an
opt-in alternative to ENet for same-machine zone-to-zone traffic.
ENet remains the default and stays for zone-to-player. Gated by
`#ifdef UNIX_ENABLED`. The RTT-derived adaptive timeout already
fixes the 144-entity burst under ENet fragmentation. UDS removes
fragmentation overhead entirely but is only relevant for same-machine
deployments — unnecessary mass until fan-out measurement (item 7)
shows ENet is the bottleneck.

## ~~Editor zone visualizer~~ DEFERRED

Hilbert band overlay, entity count per zone, migration arrows in the
3D viewport. Unnecessary mass before the VR smoke test passes (item 1).
Revisit if debugging items 5+ becomes painful without visualization.

## ~~Editor multiplayer_fabric awareness~~ DEFERRED

Making the editor understand zones, migration state, or CH_INTEREST
routing. The demo runs headless zone servers plus a PCVR client; the
editor does not participate in the fabric. Godot's existing "Run
Multiple Instances" covers the multi-process case. Adding editor
integration is maintenance surface area that breaks across Godot
versions and solves no current risk.

## Reach 1,000 concurrent players across one fabric


Sizing: 1,000 players x 56 entities = 56,000; at 1,800 per zone,
32 zones; 7 zones per 8-core machine, 5 machines, 63,000 capacity,
1,125 players with 12.5% headroom.

**Risk retired:** the full stack holds under production-scale load.

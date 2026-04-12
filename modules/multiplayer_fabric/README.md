# multiplayer_fabric

A Godot C++ module that lets many servers share one seamless world. When a player or entity crosses the invisible border between two server zones, it hands off with zero jitter — no pop, no rubber-banding. The math that makes this safe is formally proved in Lean 4 (see `thirdparty/predictive_bvh/`).

See [CONCEPT_FABRIC.md](CONCEPT_FABRIC.md) for the demo vision and [CONTRIBUTING.md](CONTRIBUTING.md) for build commands and zone CLI.

## The big picture

```
Player client
     │  FabricMultiplayerPeer (GDScript/C++ API)
     │
     ▼
Zone server A ──── ENet ────▶ Zone server B
     │                              │
     │   STAGING migration          │
     │   (proved: exactly one       │
     │    owner at all times)       │
     │                              │
     └── FabricZone (headless) ─────┘
              │
              ▼
        PredictiveBVH
        (spatial oracle: who can see whom?)
```

A **zone** is a headless Godot process that owns a spatial region. Zones connect to each other over ENet. When an entity needs to move to the adjacent zone, it goes through a **STAGING** state — both zones are aware of it until the handoff commits. This protocol is formally proved to never produce two simultaneous owners.

The **PredictiveBVH** answers one question each tick: "given this entity's position, velocity, and acceleration, which other entities will it interact with in the next δ ticks?" This lets zones send interest updates without querying every entity every frame.

## Usage in GDScript

```gdscript
# Connect a client to a zone fabric
var peer := FabricMultiplayerPeer.new()
peer.game_id = "my_game"
peer.create_client("127.0.0.1", 17500)
multiplayer.multiplayer_peer = peer
```

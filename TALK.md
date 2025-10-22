# The Invisible Artist:

## How Computers Paint Through Glass Without Going Crazy

### A TED Talk on Order-Independent Transparency

Imagine you're a painter, commissioned to create a masterpiece of see-through objects. A vase filled with colorful liquids, behind it a crystal sculpture, and floating between them, wisps of colored smoke. But your canvas isn't wood or paper—it's a computer screen. And your task is to make all these translucent objects render perfectly, without the computer getting confused about which color goes where.

This is the challenge of [Order-Independent Transparency (OIT)](https://en.wikipedia.org/wiki/Order-Independent_Transparency), and today I want to tell you about a revolutionary approach that Godot Engine uses to solve this digital painting paradox.

## The Transparency Problem

In traditional computer graphics, we render objects back-to-front. If I have a red sphere behind a blue glass cube, I draw the sphere first, then the glass cube on top. But what happens when we have _many_ translucent objects, all interpenetrating in complex ways?

```
Traditional Graphics: ❌
Red Sphere → Blue Cube → Green Cylinder
Power Pole  ← Glass Tube ← Tree Branch

The Problem: 🔄
- Fragments arrive in ANY order
- We don't know which is "behind" which
- Mixing colors becomes garbage
```

The result? Ugly sorting artifacts, wrong colors, and games that look like they're running on 1990s hardware.

## The Failed Solutions

Game developers have tried many approaches, each with fatal flaws:

### Depth Peeling 🔪

Like peeling an onion, layer by layer:

-   Pass 1: Render backmost layer
-   Pass 2: Find next layer behind that
-   Pass 3: Find layer behind _that_

Problems:

-   Needs as many passes as layers ➡️ Slow for many translucent objects
-   Can't handle arbitrary complexity

### Weighted Blended OIT ⚖️

Give each fragment a "weight" based on depth:

```
Far fragment: weight = 0.00001
Near fragment: weight = 1.0
```

Problems:

-   Impossible to create perfectly clear glass
-   Blending artifacts when depths are similar

### Conventional Linked Lists 🔗

Store ALL fragments in GPU memory:

-   Fragment arrives ➡️ Add to global list
-   After all rendering ➡️ Sort and blend

Problems:

-   Massive memory usage for complex scenes
-   Sorting millions of fragments = slow

## The Genius Solution: Tile-Based Per-Pixel Linked Lists

Godot's approach divides the screen into **small tiles** (16×16 pixels each). Each tile maintains its own fragment storage, dramatically reducing memory needs.

```
Screen: 1920×1080 pixels
Tile Size: 16×16
Tiles per row: 120 tiles
Total tiles: 120 × 68 = ~8,16 tiles

Per-tile memory: ~1KB instead of 1GB+ globally!
```

### Phase 1: Collection - "Storing the Chaos"

When a translucent fragment (piece of transparent surface) hits a pixel within a tile:

```glsl
// In Material Shader - Collection Phase
uint tile_idx = calculate_tile_index(pixel_x, pixel_y);
uint list_position = atomic_counter_increment(tile_counter[tile_idx]);

fragments[list_position] = FragmentData(next: tile_head[tile_idx],
                                        depth: encoded_z,
                                        color: rgba_packed);
tile_head[tile_idx] = list_position;
```

**What happens here?**

-   **Atomic Counter**: Thread-safe position allocation (no race conditions!)
-   **Linked List**: Each fragment points to the previous one in the tile
-   **Per-Tile Storage**: Only fragments hitting THIS 16×16 region

### Phase 2: Resolve - "Putting the Puzzle Together"

For each pixel that needs transparency:

```glsl
// In Resolve Shader
FragmentData collected[64]; // Collect up to 64 fragments
int num_frags = 0;

// Walk linked list, collect fragments
uint current = tile.fragment_head;
while (current != ~0u && num_frags < 64) {
    collected[num_frags] = frags[current];
    num_frags++;
    current = frags[current].next;
}

// BUBBLE SORT - Most Expensive Part!
for (int i = 0; i < num_frags - 1; ++i) {
    for (int j = 0; j < num_frags - i - 1; ++j) {
        if (collected[j].depth_packed > collected[j + 1].depth_packed) {
            // Swap fragments to sort by depth
        }
    }
}

// Blend back-to-front (far to near)
for (int i = num_frags - 1; i >= 0; --i) {
    vec4 frag_color = unpack_color(collected[i]);
    accum = blend(accum, frag_color); // Standard alpha blending
}
```

**Why bubble sort?**

-   ✓ Simple to implement in GLSL
-   ✓ Works perfectly for small lists (64 fragments max)
-   ✓ Deterministic results
-   ✗ O(n²) complexity - but n is small!

## Visualizing the Magic

```
Frame Time: 16.67ms @ 60fps

╔══════════════════════════════════════════════════╗
║ Rendering Pipeline                              ║
║                                                  ║
║  ┌─────────────────────────────────┐            ║
║  │ Opaque Pass                    │            ║
║  │ (trees, buildings, etc.)       │            ║
║  └─────────────────────────────────┘            ║
║           │                                      ║
║           ▼                                      ║
║  ┌─────────────────────────────────┐            ║
║  │ Transparent Collection         │            ║
║  │ ⚫⚫⚫⚫ (fragments scatter      │            ║
║  │      randomly to tiles)        │            ║
║  └─────────────────────────────────┘            ║
║           │                                      ║
║           ▼                                      ║
║  ┌─────────────────────────────────┐            ║
║  │ Transparent Resolve           │            ║
║  │ 🧩 Sort each pixel's puzzle   │            ║
║  │   and blend correctly          │            ║
║  └─────────────────────────────────┘            ║
║                                                  ║
║ Total: Perfect transparency! 🌈                ║
╚══════════════════════════════════════════════════╝
```

## The Math That Makes It Work

### Position Encoding

Fragment depth is packed as 32-bit integers for faster sorting:

```glsl
// Encode floating-point depth as sortable integer
uint pack_depth(float depth) {
    return floatBitsToUint(depth);
}
// But fragments can be equal depth!
```

### Blending Equation

Each fragment blends back-to-front:

```
C_out = C_accum × (1 - A_frag) + C_frag × A_frag
A_out = 1 (final output always opaque)
```

This is why we sort near-to-far internally, then blend far-to-near.

## Performance Trade-offs

**Pros:**

-   ✅ Handles unlimited layers per pixel
-   ✅ Memory efficient (tile-local storage)
-   ✅ Perfect accuracy for glass/clear materials
-   ✅ Works with arbitrary geometry complexity

**Cons:**

-   ❌ Bubble sort limits complex pixels to 64 fragments
-   ❌ Additional GPU passes = more draw calls
-   ❌ Memory allocation per frame (more than traditional)

**Real-world performance:**

-   Average scene: ~2-3 fragments per transparent pixel
-   Bubble sort on 3 items = instant
-   Total overhead: ~10-15% GPU time

## Beyond Gaming: Real-World Impact

This technology enables:

-   **Medical Visualization**: Semi-transparent organ layers, blood vessels through tissue
-   **Architectural Design**: Glass buildings, water features, atmospheric effects
-   **Scientific Simulation**: Fluid dynamics, molecular visualization
-   **VR/AR**: Frosted glass interfaces, holographic projections

## The Future and Beyond

As GPUs evolve, we see optimizations:

-   **GPUs with hardware sorting** (AMD RDNA3+, NVIDIA Ampere+)
-   **Variable rate shading** - lower quality transparency in distance
-   **Temporal accumulation** - reuse work across frames

## Conclusion

Order-Independent Transparency transforms how computers render the translucent world. By cleverly partitioning the screen into tiles and maintaining per-pixel fragment lists, we've created a system that can handle the complexity of real glass, water, and atmosphere.

This isn't just about prettier games—it's about enabling new forms of visual communication. When we look at the world through truly transparent digital lenses, we see possibilities we never imagined.

The invisible artist can finally paint without going crazy. And that's pretty transparent. 😉

---

**Q&A**

_Q: Why not just use Weighted OIT for everything?_
A: WBOIT can't create perfectly transparent objects. It approximates blending but glass will never be "perfect" glass.

_Q: What if I have hundreds of layers?_
A: This system caps at 64 fragments per pixel per tile. For extreme cases, you'd need depth peeling or multi-pass approaches.

_Q: This seems complicated..._
A: It is! But 15 years ago, this was PhD-level research. Today it's production-ready for game engines.

_Q: Does this work on mobile?_
A: Sort-of. Mobile GPUs support storage buffers but sorting 64 items might be too slow. Use WBOIT fallback.

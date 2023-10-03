# CASSIE task graph

```mermaid
graph LR
    E[3D Path] --> F[Beautify]
    F[Beautify] --> A[Godot tetrahedralize] 
    A[Godot tetrahedralize] -- used by --> B[multipolygon triangulator]
    B[multipolygon triangulator] -- refined --> G[Instrinsic triangle remeshing]
```
# Robust Skin Weights Transfer via Weight Inpainting

![Teaser](https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/teaser.jpg)
Sample code for the Siggraph Asia 2023 Technical Communications paper - [Robust Skin Weights Transfer via Weight Inpainting](https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/index.html)

## Dependencies

Python bindings for [libigl](https://github.com/libigl/libigl-python-bindings)

## Running

### Simple Transfer

```python
brew install micromamba
/opt/homebrew/opt/micromamba/bin/micromamba shell init -s zsh -p ~/micromamba
source ~/.zshrc
micromamba create --name robust_weight_transfer -c conda-forge python=3.10
micromamba activate robust_weight_transfer
arch -arm64 pip3 install libigl robust_laplacian
python src/main.py
```

This will perform a simple transfer of weights from a sphere to the plane above it.

![SphereToPlane](imgs/SphereToPlane.png)

However, the code contains the full implementation of the method, and you can swap
the meshes with any other meshes and load the source skinning weights.

### Body to garment transfer

(Coming soon) Load fbx files of a body and cloth meshes. Do the transfer from
the body to cloth and write the result of the transfer into another fbx that can
be loaded in other 3D software (Blender, Unreal, etc.).

## Other 3rd party implementations

Blender addon (by Sent From Space) - https://sentfromspacevr.gumroad.com/l/robust-weight-transfer

## Shrinkwrap operator

iFire — Today
I have that operator.
def find_closest_point_on_surface(points, mesh_vertices, mesh_triangles):
"""
Args:
points (np.ndarray): An array of shape (#points, 3), where each row represents the coordinates of a point in 3D space.
mesh_vertices (np.ndarray): An array of shape (#mesh_vertices, 3), representing the vertices of the target mesh surface.
mesh_triangles (np.ndarray): An array of shape (#mesh_triangles, 3), where each row contains indices that correspond to the vertices forming a triangle on the target mesh surface.

    Returns:
        tuple: A tuple containing four elements:
            - smallest_squared_distances (np.ndarray): An array of shape (#points,), which holds the smallest squared distances from each input point to the closest point on the target mesh surface.
            - primitive_indices (np.ndarray): An array of shape (#points,), indicating the index of the triangle in `mesh_triangles` where the closest point was found for each input point.
            - closest_points (np.ndarray): An array of shape (#points, 3), representing the coordinates of the closest points on the target mesh surface for each input point.
            - barycentric_coordinates (np.ndarray): An array of shape (#points, 3), providing the barycentric coordinates of each closest point relative to the vertices of the triangle it lies on.
    """

I feel like I have all the pieces, but missing something obvious.
given barycentric_coordinates we can find the normal on the triangle index / triangle positions

janie bean — Today

A shrinkwrap operator takes verts and snaps them to the closest point on a mesh + offset \* normal, right?
the only part i don't know how to do is that "find closest point on mesh" operation
and then...find the normal of it

janie bean — Today

Yeah, so like, to create a "cage" for a mesh you basically just take the position of each vert and add the normal \* an offset
and a shrinkwrap basically takes a mesh and snaps it to the cage of another mesh

if find_closest_point_on_surface can get the normal of that point, then can it also get the weight?

Chev — Today

you can use the index of the triangle to look up other information
it's the backend of robust weight transfer. trying to port it from python to c++

janie bean — Today

wait, so what you're asking for is something that will deform a mesh kinda like a shrinkwrap modifier, but keep some of the 3D details of the clothing you're modeling?

## Cite

If you use this code for an academic publication, cite it as:

```bib
@inproceedings{abdrashitov2023robust,
author = {Abdrashitov, Rinat and Raichstat, Kim and Monsen, Jared and Hill, David},
title = {Robust Skin Weights Transfer via Weight Inpainting},
year = {2023},
isbn = {9798400703140},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3610543.3626180},
doi = {10.1145/3610543.3626180},
booktitle = {SIGGRAPH Asia 2023 Technical Communications},
articleno = {25},
numpages = {4},
location = {<conf-loc>, <city>Sydney</city>, <state>NSW</state>, <country>Australia</country>, </conf-loc>},
series = {SA '23}
}
```

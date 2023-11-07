# Dem Bones

This repository contains an implementation of [Smooth Skinning Decomposition with Rigid Bones](http://binh.graphics/papers/2012sa-ssdr/),
an automated algorithm to extract the *Linear Blend Skinning* (LBS) with bone transformations from a set of example meshes.
*Skinning Decomposition* can be used in various tasks:

- converting any animated mesh sequence, e.g. geometry cache, to LBS, which can be replayed in popular game engines,
- solving skinning weights from shapes and skeleton poses, e.g. converting blendshapes to LBS,
- solving bone transformations for a mesh animation given skinning weights.

This project is named after "The Skeleton Dance" by Super Simple Songs.

## Contents
- `include/DemBones`: C++ header-only core library using [Eigen](http://eigen.tuxfamily.org) and [OpenMP](https://www.openmp.org/). Check out the documentations in [docs/index.html](docs/index.html).
- `data`: input/output test data for the command line tool. Run and check out the scripts `run.bat` (Windows) or `./run.sh` (Linux/MacOS).
- `data_gltf`: input/output test data in gltf2 format.

## References

If you use the library or the command line tool, please cite the paper:

> *Binh Huy Le* and *Zhigang Deng*. **[Smooth Skinning Decomposition with Rigid Bones](http://binh.graphics/papers/2012sa-ssdr/)**. ACM Transactions on Graphics 31(6), Proceedings of ACM SIGGRAPH Asia 2012.

BibTeX:

```
@article{LeDeng2012,
    author = {Le, Binh Huy and Deng, Zhigang},
    title = {Smooth Skinning Decomposition with Rigid Bones},
    journal = {ACM Trans. Graph.},
    volume = {31},
    number = {6},
    year = {2012}
}
```

The skinning weights smoothing regularization was published in the paper:

> *Binh Huy Le* and *Zhigang Deng*. **[Robust and Accurate Skeletal Rigging from Mesh Sequences](http://binh.graphics/papers/2014s-ske/)**. ACM Transactions on Graphics 33(4), Proceedings of ACM SIGGRAPH 2014.

## Authors

<b>Search for Extraordinary Experiences Division (SEED) - Electronic Arts <br> http://seed.ea.com</b><br>
We are a cross-disciplinary team within EA Worldwide Studios.<br>
Our mission is to explore, build and help define the future of interactive entertainment.</p>

Dem Bones was created by Binh Le (ble@ea.com). The [logo](logo/DemBones.png) was designed by Phuong Le.

## Licenses

- The source code, including `include/DemBones` uses *BSD 3-Clause License* as detailed in [LICENSE.md](LICENSE.md)
- The pre-compiled command line tool `bin/DemBones`(`.exe`) uses third party libraries: Eigen with licenses in [3RDPARTYLICENSES.md](3RDPARTYLICENSES.md)


# Mesh Self Intersection

This package provides a PyTorch module used to detect self-intersections in a
triangle mesh.


## Table of Contents
  * [License](#license)
  * [Description](#description)
  * [Installation](#installation)
  * [Examples](#examples)
  * [Citation](#citation)
  * [Contact](#contact)

## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and
conditions](https://github.com/vchoutas/smplx/blob/master/LICENSE) and any
accompanying documentation before you download and/or use the SMPL-X/SMPLify-X
model, data and software, (the "Model & Software"), including 3D meshes, blend
weights, blend shapes, textures, software, scripts, and animations. By
downloading and/or using the Model & Software (including downloading, cloning,
installing, and any other use of this github repository), you acknowledge that
you have read these terms and conditions, understand them, and agree to be bound
by them. If you do not agree with these terms and conditions, you must not
download and/or use the Model & Software. Any infringement of the terms of this
agreement will automatically terminate your rights under this
[License](./LICENSE).


## Description

This repository provides a PyTorch wrapper around a CUDA kernel that implements
the method described in [Maximizing parallelism in the construction of BVHs,
octrees, and k-d trees](https://dl.acm.org/citation.cfm?id=2383801). More
specifically, given input meshes with the *same* number of faces, it builds a
BVH tree for each one and queries it for self-intersections. Moreover, we
provide a conical distance field based loss for resolving the interpenetrations.


## Installation

Before installing anything please make sure to set the environment variable
*$CUDA_SAMPLES_INC* to the path that contains the header `helper_math.h`.
To install the module you can:

1. Clone this repository and install it using the *setup.py* script:
```Shell
git clone https://github.com/vchoutas/torch-mesh-isect
python setup.py install
```
2. Install from PyPi the pre-compiled version for 1.1.0 and CUDA 10.0 simply run:
```Shell
pip install torch-mesh-isect
```

3.  To install the module and the rendering dependencies run:
```Shell
pip
install torch-mesh-isect[all]
```


## Examples

* [Collision Detection](./examples/detect_and_plot_collisions.py): Given an
  input mesh file, detect and plot all the collisions.
* [Batch Collision resolution](./examples/batch_smpl_untangle.py): Same as the
  second example, except that a batch of examples is processed simultaneously

## Dependencies

1. [PyTorch](https://pytorch.org/)

### Optional Dependencies

1. [Trimesh](https://trimsh.org/) for loading triangular meshes
2. [Pyrender](https://pyrender.readthedocs.io/) for visualization

## Citation

If you find this code useful in your research then please cite:

```
@inproceedings{SMPL-X:2019,
title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
booktitle = {Proceedings IEEE Conf.
on Computer Vision and Pattern Recognition (CVPR)}, year = {2019} }
```

```
@article{Tzionas:IJCV:2016, title = {Capturing Hands in Action using Discriminative Salient Points and Physics Simulation},
author = {Tzionas, Dimitrios and Ballan, Luca and Srikantha, Abhilash and Aponte, Pablo and Pollefeys, Marc and Gall, Juergen},
journal = {International Journal of Computer Vision (IJCV)},
volume = {118},
number = {2},
pages = {172--193},
month = jun,
year = {2016},
url = {https://doi.org/10.1007/s11263-016-0895-4}, month_numeric
= {6} }
```

```
@inproceedings{Karras:2012:MPC:2383795.2383801,
author = {Karras, Tero},
title = {Maximizing Parallelism in the Construction of BVHs, Octrees, and K-d Trees},
booktitle = {Proceedings of the Fourth ACM SIGGRAPH / Eurographics
Conference on High-Performance Graphics},
series = {EGGH-HPG'12},
year = {2012},
isbn = {978-3-905674-41-5},
location = {Paris, France},
pages = {33--37},
numpages = {5},
url = {https://doi.org/10.2312/EGGH/HPG12/033-037}, doi =
{10.2312/EGGH/HPG12/033-037},
acmid = {2383801},
publisher = {Eurographics
Association},
address = {Goslar Germany, Germany}, }
```

## Contact
For questions about the code, please contact [Vassilis Choutas](vassilis.choutas@tuebingen.mpg.de).

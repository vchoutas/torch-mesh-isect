# Detecting & Penalizing Mesh Intersections

This package provides a PyTorch module that can efficiently (1) detect and (2) penalize (self-)intersections for a triangular mesh.


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
conditions](https://github.com/vchoutas/torch-mesh-isect/blob/master/LICENSE) and any
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
specifically, given an input mesh it builds a
BVH tree for each one and queries it for self-intersections. Moreover, we
provide a conical 3D distance field based loss for resolving the interpenetrations, as in [Capturing Hands in Action using Discriminative Salient Points and Physics Simulation](https://doi.org/10.1007/s11263-016-0895-4).

Please note that in the current implementation, for batching one needs to provide meshes with the *same* number of faces. Moreover, the code by default works for self-penetrations of a body mesh. The module can be used also for inter-penetrations of different meshes - for this the easiest and naive approach (without additional bookkeeping) is to fuse all meshes in a single mesh and treat inter-penetrations as self-penetrations.


## Installation

Before installing anything please make sure to set the environment variable
*$CUDA_SAMPLES_INC* to the path that contains the header `helper_math.h`, which
can be found in the repo [CUDA Samples repository](https://github.com/NVIDIA/cuda-samples).
To install the module run the following commands:  

**1. Clone this repository**
```Shell
git clone https://github.com/vchoutas/torch-mesh-isect
cd torch-mesh-isect
```
**2. Install the dependencies**
```Shell
pip install -r requirements.txt 
```
**3. Run the *setup.py* script**
```Shell
python setup.py install
```

## Examples

* [Collision Detection](./examples/detect_and_plot_collisions.py): Given an
  input mesh file, detect and plot all the collisions. Use:
  ```Shell
  python examples/detect_and_plot_collisions.py PATH_TO_MESH
  ```
* [Batch Collision resolution](./examples/batch_smpl_untangle.py):  Resolve self-penetrations for a batch of body models. To run use:
  ```Shell
  WEIGHT=0.001
  python examples/batch_smpl_untangle.py --coll_loss_weight=$WEIGHT --model_folder=$MODEL_PARENT_FOLDER --part_segm_fn=$PATH_part_segm_fn 
  --param_fn PKL_FN1 PKL_FN2 ... PKL_FNN  
  ```
  where `PKL_FN*` are the filenames of the .pkl files that [can be downloaded here](https://owncloud.tuebingen.mpg.de/index.php/s/bEKMdqf5WbN4MnH) and contain the parameters for each body model. 
  
  For `batch_smpl_untangle`:  
  - To download the SMPL body model  please see the directions for [downloding](https://github.com/vchoutas/smplx/blob/master/README.md#downloading-the-model) and for specific [placing in folder structure and renaming](https://github.com/vchoutas/smplx/blob/master/README.md#model-loading).
  - The file for the `part_segm_fn` argument for SMPL can be downloaded [here](https://owncloud.tuebingen.mpg.de/index.php/s/jHdgwkREzS43rjN).

## Dependencies

1. [PyTorch](https://pytorch.org)

### Example dependencies

1. [SMPL-X](https://github.com/vchoutas/smplx)

### Optional Dependencies

1. [Trimesh](https://trimsh.org) for loading triangular meshes
2. [Pyrender](https://pyrender.readthedocs.io) for visualization

The code has been tested with Python 3.6, CUDA 10.0, CuDNN 7.3 and PyTorch 1.0.

## Citation

If you find this code useful in your research please cite the relevant work(s) of the following list, for detecting and penalizing mesh intersections accordingly:

```
@inproceedings{Karras:2012:MPC:2383795.2383801,
    author = {Karras, Tero},
    title = {Maximizing Parallelism in the Construction of BVHs, Octrees, and K-d Trees},
    booktitle = {Proceedings of the Fourth ACM SIGGRAPH / Eurographics Conference on High-Performance Graphics},
    year = {2012},
    pages = {33--37},
    numpages = {5},
    url = {https://doi.org/10.2312/EGGH/HPG12/033-037}, 
    doi = {10.2312/EGGH/HPG12/033-037},
    publisher = {Eurographics Association}
}
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
    url = {https://doi.org/10.1007/s11263-016-0895-4}, 
    month_numeric = {6} 
}
```

This repository was originally developed for SMPL-X / SMPLify-X (CVPR 2019), you might be interested in having a look: [https://smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de).


## Contact
The code of this repository was implemented by [Vassilis Choutas](vassilis.choutas@tuebingen.mpg.de).

For questions, please contact [smplx@tue.mpg.de](smplx@tue.mpg.de). 

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](ps-licensing@tue.mpg.de). Please note that the method for this component has been [patented by NVidia](https://patents.google.com/patent/US9396512B2/en) and a license needs to be obtained also by them.

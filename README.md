## ElasticIsoLib-3D
GPU-based elastic isotropic wave-equation modeling and inversion library

##DESCRIPTION
Note use cmake 3.14

##COMPILATION

When the package is cloned, run the following command once:
```
git submodule update --init --recursive -- elastic_iso_lib_3d/external/ioLibs
git submodule update --init --recursive -- elastic_iso_lib_3d/external/pySolver

```

To build library run:
```
cd build

cmake -DCMAKE_INSTALL_PREFIX=installation_path -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc ../elastic_iso_lib_3d/

make install

```

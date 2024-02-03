# ReflectanceRCWA

The repository is locatd in [RCWA](https://github.com/Hoang-LamPham/RCWA/tree/main) and the notebook can be found in [Example](https://github.com/Hoang-LamPham/RCWA/tree/main/Example) 

## Introduction

We present an efficient RCWA simulation of 3D multilayers with bottom homogeneous layers.

In practical applications such as biosensing, optical diffusers, solar cells, photodetectors, and nanostructures, which are composed of top gratings and bottom homogeneous layers for tuning the efficiency of optical devices. Simulating such multilayers involves computing s-matrices of gratings and homogeneous layers, and then combining these multiple s-matrices globally to quantify expected optical responses.

Although the computation of a homogeneous layer is simple without solving eigen decomposition, handling a stack of homogeneous layers could be a numerical issue in 3D structures, as these layers involve expensive matrix algebra to connect to grating layers.

To alleviate this issue, our approach introduces a vector-based formation to circumvent the large matrix computation of homogeneous layers. The application of the formation, along with bottom-up construction, improved numerical efficiency. The notebook is used to illustrate a demonstration of a 3D plasmonic structure.


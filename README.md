# ReflectanceRCWA

The repo is in [ScatteringM](https://github.com/Hoang-LamPham/ScatteringM/) and the notebook can be found in [RCWA module](https://github.com/Hoang-LamPham/ScatteringM/tree/main/1RCWA) of the project.

### Introduction
We present an efficient RCWA simulation of 3D multilayers with bottom homogeneous layers.

In practical applications of biosensing, optical diffuser, solar cells, photo-dectectors, nanostructures are composed of top gratings and bottom homogeneous layers 
for tuning the efficiency of the optical devices. Simulating such multilayers involves computing s-matrices of gratings and homogeneous layers, then combining these 
multiple s-matrices for global one to quantify expected optical responses.

Although the computation of a homogeneous layer is simple without solving eigen decomposition, handling stack of homogeneous layers could be a numerical issue in 
3D-structures, as these layers involve expensive matrix algebra in order to connect to grating layers.

To alleviate the issue, our approach introduces vector-based formation to circumvent the large matrix computation of homogeneous layers. 
The application of the formation along with bottom-up construction improved numerical efficiency. The notebook is used to illustrate a demonstration 
of 3D plasmonic structure

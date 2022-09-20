# mGPUKDE_tiled
Multi-GPU-parallel and tile-based kernel density estimation for large-scale spatial point pattern analysis
## Dependencies
1. CUDA 11.7: https://developer.nvidia.com/cuda-11-7-0-download-archive.
2. GDAL: https://gdal.org/index.html.
## IDE
Visual Studio 2022.
## How to configure the project before building in Visual Studio 2022 (Community Version)?
### Step 1. Open the solution by opening the .sln file
2. Edit project properties as below:
Project --> Properties --> Configuration Properties 
...VC++ Directories; edit "Include Directories" and "Library Directories" to point to the GDAL include and lib folders, respectively.
...CUDA C/C++; configure "CUDA Toolkit Custom Dir"
## How to run the program?
A complied mGPUKDE_tiled.exe file is provided in the x64\Debug folder. Below are two example commands to run the program:
1. run in mode 0 (data are generated on the fly):
mGPUKDE_tiled.exe 0 100 1 0.1 0 1 0 1 0 12 2 density_cpu.tif density_gpu.tif 0
2. run in mode 1 (data are read from files):
mGPUKDE_tiled.exe 1 pntsRedwood.csv redwood_edgecorrection.tif redwood_densityestimation.tif 2 1 0 1 0 12 2 density_cpu.tif density_gpu.tif 0

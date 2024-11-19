# openfish

Openfish is a submodule used in [slorado](https://github.com/BonsonW/slorado) for performing beamsearch decoding. 
The CPU implementation written in C was adopted from the C++ beamsearch implementation in [ONT Dorado](https://github.com/nanoporetech/dorado) which is licensed under the [Oxford Nanopore Technologies PLC. Public License Version 1.0](https://github.com/nanoporetech/dorado/blob/release-v0.8/LICENCE.txt). 
Then based on that C implementation, we implemented the GPU version for NVIDIA GPUs using CUDA C and for AMD GPUs using HIP C.   

This is in early stages of development and thus the API not at all stable (and thus not documented).

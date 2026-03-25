# openfish

*openfish* is a library for CRF-CTC beam-search decoding used in nanopore basecalling. It supports CPU, NVIDIA GPU (CUDA) and AMD GPU (ROCm/HIP).

The CPU implementation was adopted from the C++ beam-search implementation in [ONT Dorado](https://github.com/nanoporetech/dorado) (licensed under the [Oxford Nanopore Technologies PLC. Public License Version 1.0](https://github.com/nanoporetech/dorado/blob/release-v0.8/LICENCE.txt)) and re-written in C. GPU backends were then built on top of that C implementation.

*openfish* is used as a submodule in [slorado](https://github.com/BonsonW/slorado). If you are a user who wants to basecall nanopore reads, please visit [slorado](https://github.com/BonsonW/slorado) instead. This repository is intended for developers who want to integrate openfish decoding into their own applications. Please note that openfish is still in early phases, so the API could change in future versions.

## Table of Contents

- [Building](#building)
  - [CPU-only](#cpu-only)
  - [NVIDIA GPU (CUDA)](#nvidia-gpu-cuda)
  - [AMD GPU (ROCm)](#amd-gpu-rocm)
  - [Optional build flags](#optional-build-flags)
- [Usage](#usage)
  - [Integrating as a library](#integrating-as-a-library)
  - [API overview](#api-overview)
  - [Quick run / validation](#quick-run--validation)
- [Acknowledgements](#acknowledgements)

## Building

### CPU-only

Building *openfish* requires GCC and standard development tools (`make`, `ar`).

```sh
git clone https://github.com/hasindu2008/openfish
cd openfish
make
```

This produces `lib/libopenfish.a` for static linking.

### NVIDIA GPU (CUDA)

Requires the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (tested with CUDA 11+).

```sh
make cuda=1
```

The CUDA root is auto-detected at `/usr/local/cuda`. Override with `CUDA_ROOT=/path/to/cuda make cuda=1`.
To target a specific GPU architecture, pass the `nvcc` architecture flag:

```sh
make cuda=1 CUDA_ARCH="-gencode arch=compute_80,code=sm_80"
```

### AMD GPU (ROCm)

Requires [ROCm](https://rocm.docs.amd.com/) (tested with ROCm 5+).

```sh
make rocm=1
```

The ROCm root is auto-detected at `/opt/rocm`. Override with `ROCM_ROOT=/path/to/rocm make rocm=1`.
To target a specific GPU architecture, pass the `hipcc` architecture flag:

```sh
make rocm=1 ROCM_ARCH="--offload-arch=gfx90a"
```

### Optional build flags

| Flag | Description |
|------|-------------|
| `debug=1` | Enable debug output and OpenMP support in the test binary |
| `asan=1` | Enable AddressSanitizer (`-fsanitize=address`) |
| `bench=1` | Enable internal benchmarking output |

Flags can be combined, e.g. `make cuda=1 debug=1`.

## Usage

### Integrating as a library

Include `<openfish/openfish.h>` in your C (or C++) source and link against `lib/libopenfish.a`:

```sh
# static linking (CPU build)
gcc [OPTIONS] -I path/to/openfish/include your_program.c \
    path/to/openfish/lib/libopenfish.a -lz -lm -lpthread -o your_program

# static linking (CUDA build) — also link the CUDA runtime
gcc [OPTIONS] -I path/to/openfish/include your_program.c \
    path/to/openfish/lib/libopenfish.a -lz -lm -lpthread \
    -L/usr/local/cuda/lib64 -lcudart_static -lrt -ldl -o your_program

# static linking (ROCM build)
gcc [OPTIONS] -I path/to/openfish/include your_program.c \
    path/to/openfish/lib/libopenfish.a -lz -lm -lpthread \
    -L/opt/rocm/lib -lamdhip64 -lrt -ldl -o your_program

```

*path/to/openfish/* is the absolute or relative path to the cloned repository.

### API overview

Please see [here](docs/api.md)

### Quick run / validation

The scripts below download a set of pre-computed blobs and compare *openfish* output against them. They require the binary to be built with `debug=1`.

**CPU:**

```sh
make debug=1
scripts/cpu_quick_run.sh fast   # fast model  (state_len=3, C=64)
scripts/cpu_quick_run.sh hac    # hac model   (state_len=4, C=256)
scripts/cpu_quick_run.sh sup    # sup model   (state_len=5, C=1024)
```

**GPU (CUDA):**

```sh
make cuda=1 debug=1
scripts/gpu_quick_run.sh fast
scripts/gpu_quick_run.sh hac
scripts/gpu_quick_run.sh sup
```

**GPU (ROCm):**

```sh
make rocm=1 debug=1
scripts/gpu_quick_run.sh fast
scripts/gpu_quick_run.sh hac
scripts/gpu_quick_run.sh sup
```

## Acknowledgements

The CPU beam-search implementation is derived from the C++ implementation in [ONT Dorado](https://github.com/nanoporetech/dorado).

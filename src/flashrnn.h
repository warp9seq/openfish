// Copyright 2024 NXAI GmbH, All Rights Reserved
// Author: Korbinian Poeppel
// Adapted from the haste library
//
// See:
// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#pragma once

#include "util/util.h"
#include <cublas_v2.h>
#include <openfish/openfish.h>

class ForwardPass {
public:
  // training: `true` if the caller intends to perform a backward pass to
  // compute gradients. batch_size: the number of training/inference inputs
  // provided in each tensor. input_size: the dimension of each input vector.
  // hidden_size: the expected dimension of each output vector.
  // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
  ForwardPass(const bool training, const int batch_size, const int hidden_size,
              const int num_heads, const cublasHandle_t &blas_handle,
              const cudaStream_t &stream = 0);

  // Releases internal resources.
  // Blocks until all iterations have completed executing on the GPU.
  ~ForwardPass();

  // Set internal values for single forward / backward
  void Set(const bool training, const int batch_size, const int hidden_size,
           const int num_heads, const cublasHandle_t &blas_handle,
           const cudaStream_t &stream = 0);

  int Run(const int steps, const FLASHRNN_DTYPE_R *R, const FLASHRNN_DTYPE_B *b,
          const FLASHRNN_DTYPE_W *x, FLASHRNN_DTYPE_S *s, FLASHRNN_DTYPE_G *g_r,
          FLASHRNN_DTYPE_G *g_i, FLASHRNN_ACC_DTYPE *gate_buffer);

private:
  struct private_data;
  private_data *data_;
};

// class BackwardPass {
// public:
//   // batch_size: the number of training inputs provided in each tensor.
//   // input_size: the dimension of each input vector.
//   // hidden_size: the expected dimension of each output vector.
//   // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
//   BackwardPass(const int batch_size, const int hidden_size, const int num_heads,
//                const cublasHandle_t &blas_handle,
//                const cudaStream_t &stream = 0);

//   // Set internal values for single forward / backward
//   void Set(const int batch_size, const int hidden_size, const int num_heads,
//            const cublasHandle_t &blas_handle, const cudaStream_t &stream = 0);

//   // Releases internal resources.
//   // Blocks until all iterations have completed executing on the GPU.
//   ~BackwardPass();

//   int Run(const int steps, const FLASHRNN_DTYPE_R *R_t,
//           const FLASHRNN_DTYPE_B *b, const FLASHRNN_DTYPE_S *s,
//           const FLASHRNN_DTYPE_S *ds_new, FLASHRNN_DTYPE_R *dR,
//           FLASHRNN_DTYPE_B *db, FLASHRNN_DTYPE_S *ds, FLASHRNN_DTYPE_G *g_r,
//           FLASHRNN_DTYPE_G *g_i, FLASHRNN_DTYPE_G *g_b,
//           FLASHRNN_ACC_DTYPE *d_state_buffer);

// private:
//   struct private_data;
//   private_data *data_;
// };

// class BackwardPassCut {
// public:
//   // batch_size: the number of training inputs provided in each tensor.
//   // input_size: the dimension of each input vector.
//   // hidden_size: the expected dimension of each output vector.
//   // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
//   BackwardPassCut(const int batch_size, const int hidden_size,
//                   const int num_heads, const cublasHandle_t &blas_handle,
//                   const cudaStream_t &stream = 0);

//   // Set internal values for single forward / backward
//   void Set(const int batch_size, const int hidden_size, const int num_heads,
//            const cublasHandle_t &blas_handle, const cudaStream_t &stream = 0);

//   // Releases internal resources.
//   // Blocks until all iterations have completed executing on the GPU.
//   ~BackwardPassCut();

//   int Run(const int steps, const FLASHRNN_DTYPE_R *R_t,
//           const FLASHRNN_DTYPE_B *b, const FLASHRNN_DTYPE_S *s,
//           const FLASHRNN_DTYPE_S *ds_new, FLASHRNN_DTYPE_R *dR,
//           FLASHRNN_DTYPE_B *db, FLASHRNN_DTYPE_S *ds, FLASHRNN_DTYPE_G *g_r,
//           FLASHRNN_DTYPE_G *g_i, FLASHRNN_DTYPE_G *g_bias,
//           FLASHRNN_ACC_DTYPE *d_state_buffer);

// private:
//   struct private_data;
//   private_data *data_;
// };


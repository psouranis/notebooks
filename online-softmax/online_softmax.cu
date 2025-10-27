#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <float.h>

// We need to hold the max and the normalizer to apply the operator. Align the struct to apply vector instructions (__expf)
struct __align__(8) MaxNorm {
  float m;
  float d;
};

// The oplus operator that we defined above
__device__ __forceinline__ MaxNorm reduce_md_op(MaxNorm a, MaxNorm b) {
    bool a_bigger = (a.m > b.m);
    MaxNorm bigger_m = a_bigger ? a : b;
    MaxNorm smaller_m = a_bigger ? b : a;
    MaxNorm res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void
cu_online_softmax(float * __restrict output, const float * __restrict input, int classes) {

  //  forward pointers to batch[blockIdx.x]
  // each block handles a sample in the mini-batch
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;

  typedef cub::BlockReduce<MaxNorm, THREADBLOCK_SIZE> BlockReduce;

  __shared__ typename BlockReduce::TempStorage smem;
  __shared__ MaxNorm sumAll;

  MaxNorm sumPartial;
  sumPartial.m = -FLT_MAX;
  sumPartial.d = 0.0F;

  // Partial max-sum
  #pragma unroll
  for(int offset = threadIdx.x; offset < classes; offset += THREADBLOCK_SIZE) {
    MaxNorm threadMaxNorm;
    threadMaxNorm.m = input[offset];
    threadMaxNorm.d = 1.0F;
    sumPartial = reduce_md_op(sumPartial, threadMaxNorm);
  }

  // Reduce from each block
  MaxNorm blockReduceMaxNorm = BlockReduce(smem).Reduce(sumPartial, reduce_md_op);
  if (threadIdx.x == 0)
    sumAll = blockReduceMaxNorm;
  __syncthreads();

  // Same Epilogue as previously
  float sumAllInverse = __fdividef(1.0F, sumAll.d);
  for(int offset = threadIdx.x; offset < classes; offset += THREADBLOCK_SIZE)
    output[offset] = __expf(input[offset] - sumAll.m) * sumAllInverse;
}

torch::Tensor softmax(torch::Tensor scalar_t) {

    TORCH_CHECK(scalar_t.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(scalar_t.is_contiguous(), "Input tensor must be contiguous");

    auto result = torch::zeros_like(scalar_t);

    // Launch configuration
    const int number_of_blocks = scalar_t.size(0);
    const int classes = scalar_t.size(1);

    cu_online_softmax<128><<<number_of_blocks, 128>>>(
        result.data_ptr<float>(),
        scalar_t.data_ptr<float>(),
        classes
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // Synchronize device to catch errors during kernel execution
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(err));

    return result;
}

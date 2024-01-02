#ifndef HSSINFO_HSSINFO_HPP
#define HSSINFO_HSSINFO_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/mr/new.h>
#include <thrust/mr/pool.h>

#include <cmath>
#include <iomanip>
#include <iostream>

struct printf_functor_int {
  __host__ __device__ void operator()(int x) { printf("%d ", x); }
};

struct printf_functor_float {
  __host__ __device__ void operator()(float x) { printf("%.4f ", x); }
};

template <typename T>
struct log2_functor : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T val) { return log2f(val); }
};

template <typename T>
struct double_functor : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(T val) { return val * 2; }
};

struct isUnfinished {
  __host__ __device__ bool operator()(const bool x) { return !x; }
};

struct isFinished {
  __host__ __device__ bool operator()(const bool x) { return x; }
};

//  loop1 * log2(degree1)
// +loop2 * log2(degree2)
// - loop_module * log2(degree_module)
// + connect * log2(degree_sum)
// /= degree_sum
template <typename T>
struct entropy_delta_functor : public thrust::unary_function<thrust::tuple<T, T, T, T, T, T, T, T>, T> {
  __host__ __device__ T operator()(thrust::tuple<T, T, T, T, T, T, T, T> input) {
    T entropy = (thrust::get<1>(input) * log2f(thrust::get<0>(input)) + thrust::get<3>(input) * log2f(thrust::get<2>(input)) - thrust::get<5>(input) * log2f(thrust::get<4>(input)) +
                 thrust::get<7>(input) * log2f(thrust::get<6>(input))) /
                thrust::get<6>(input);
    return entropy;
  }
};

class HSSInfo {
private:
  cudaStream_t stream{};
  cudaMemPool_t memPool{};

public:
  int nodes;
  int edges;
  thrust::device_vector<int> d_src_idx;   // edges
  thrust::device_vector<int> d_tgt_idx;   // edges
  thrust::device_vector<float> d_weights; // edges

  thrust::host_vector<thrust::host_vector<int>> h_src_locs;  // nodes
  thrust::host_vector<thrust::host_vector<int>> h_tgt_locs;  // nodes
  thrust::host_vector<thrust::host_vector<int>> h_community; // nodes
  thrust::device_vector<int> d_comity_label;                 // nodes

  thrust::device_vector<float> d_degree; // nodes
  thrust::device_vector<float> d_loop;   // nodes

  thrust::device_vector<bool> d_finished;       // edges
  thrust::device_vector<float> d_degree1;       // edges
  thrust::device_vector<float> d_degree2;       // edges
  thrust::device_vector<float> d_loop1;         // edges
  thrust::device_vector<float> d_loop2;         // edges
  thrust::device_vector<float> d_connect;       // edges
  thrust::device_vector<float> d_loop_module;   // edges
  thrust::device_vector<float> d_degree_module; // edges
  thrust::device_vector<float> d_entropy_delta; // edges

  thrust::device_vector<float> h_degree1; // edges
  thrust::device_vector<float> h_degree2; // edges
  thrust::device_vector<float> h_loop1;   // edges
  thrust::device_vector<float> h_loop2;   // edges

  float degree_sum = 0;
  int bar          = 0;

  thrust::device_vector<int> changed; //不确定位置

public:
  HSSInfo(const int &nodes, const std::vector<int> rows, const std::vector<int> cols, const std::vector<float> weights);

  void CommunityDetection();

  template <typename T>
  void Update(const T &update_idx);

  //功能性函数
  /* 获取 [first, last] 中 unfinished 的元素*/
  template <typename T, typename InputIterator, typename OutputIterator>
  T gather_unfinished(InputIterator first, InputIterator last, OutputIterator result_first);

  template <typename InputIterator, typename OutputIterator>
  void gather_idxs(InputIterator map_first, InputIterator map_last, int type, OutputIterator result_first);

  template <typename InputIterator, typename OutputIterator>
  void gather_cmty(InputIterator map_first, InputIterator map_last, OutputIterator result_first);

  template <typename T, typename InputIterator, typename OutputIterator>
  T merge_cmty_self(InputIterator skey_first, InputIterator skey_end, InputIterator tkey_first, InputIterator tkey_end, InputIterator smap_first, InputIterator tmap_first,
                    OutputIterator result_comity_first, OutputIterator result_locs_first);

  template <typename T, typename InputIterator>
  void cmty_intersection(InputIterator key1_first, InputIterator key1_end, InputIterator key2_first, InputIterator key2_end, InputIterator map1_first, InputIterator map2_first);

  template <typename T, typename InputIterator, typename OutputIterator>
  void merge_cmty(InputIterator key1_first, InputIterator key1_end, InputIterator key2_first, InputIterator key2_end, InputIterator map1_first, InputIterator map2_first,
                  OutputIterator result_map_first);

  template <typename T>
  thrust::device_ptr<T> mallocAsyncThrust(size_t n) {
    T *raw_ptr;
    cudaMallocFromPoolAsync(&raw_ptr, n * sizeof(T), this->memPool, this->stream);
    // cudaMallocManaged(&raw_ptr, n * sizeof(T));
    // cudaMalloc(&raw_ptr, n * sizeof(T));

    return thrust::device_pointer_cast(raw_ptr);
  }

  template <typename T>
  thrust::device_ptr<T> mallocAsyncThrust(size_t n, T value) {
    T *raw_ptr;
    cudaMallocFromPoolAsync(&raw_ptr, n * sizeof(T), this->memPool, this->stream);
    // cudaMallocManaged(&raw_ptr, n * sizeof(T));
    // cudaMalloc(&raw_ptr, n * sizeof(T));
    thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(raw_ptr);
    thrust::fill(d_ptr, d_ptr + n, value);
    return d_ptr;
  }

  template <typename T>
  void freeAsyncThrust(thrust::device_ptr<T> pointer) {
    cudaFreeAsync(thrust::raw_pointer_cast(pointer), this->stream);
    // cudaFree(thrust::raw_pointer_cast(pointer));
  }

  void printf_detail();
  template <typename InputIterator>
  void printf_detail(std::string name, InputIterator first, InputIterator last);

  template <typename T>
  void printf_bar(T edgeNum, T now);
};

#endif // HSSINFO_HSSINFO_HPP

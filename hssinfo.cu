#include <sys/time.h>
#include <thrust/count.h>
#include <thrust/device_delete.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_reference.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/merge.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <fstream>

#include "hssinfo.cc"
//#define debug
HSSInfo::HSSInfo(const int &nodes, const std::vector<int> rows, const std::vector<int> cols, const std::vector<float> weights) : nodes(nodes) {
  /**********  初始化数据，使用host方法进行读取数据   *************/
  std::vector<std::vector<int>> src_locs;
  std::vector<std::vector<int>> tgt_locs;
  src_locs.resize(nodes);
  tgt_locs.resize(nodes);

  std::vector<float> degree;
  std::vector<float> loop;
  std::vector<bool> finished;
  degree.resize(nodes, 0.0);
  loop.resize(nodes, 0.0);

  for (int i = 0; i < rows.size(); i++) {
    degree[rows[i]] += weights[rows[i]]; //is weight
    degree[cols[i]] += weights[cols[i]]; //is weight

    if (rows[i] == cols[i]) {
      loop[rows[i]] += weights[rows[i]]; // is weight
      finished.push_back(true);
    } else {
      finished.push_back(false);
    }

    src_locs[rows[i]].emplace_back(i);
    tgt_locs[cols[i]].emplace_back(i);
      }
  /**********  初始化数据读取完成  *************/
  this->printf_detail();
  /************  初始化 cuda stream 和 内存池  ****************/
  cudaSetDevice(0);
  cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking); //设置stream
  cudaDeviceGetDefaultMemPool(&this->memPool, 0);                  //设置内存池
  uint64_t thresholdVal = UINT64_MAX;                              //为内存池设置阈值，可以减少内存的申请与释放操作
  cudaMemPoolSetAttribute(this->memPool, cudaMemPoolAttrReleaseThreshold, (void *) &thresholdVal);
  /************  初始化 cuda stream 和 内存池 完成  ****************/

  this->edges = rows.size();

  this->h_community.resize(this->nodes);
  for (int idx = 0; idx < this->nodes; idx++) { this->h_community[idx].push_back(idx); }

  this->d_comity_label.resize(this->nodes);
  thrust::host_vector<int> h_comity_label;
  h_comity_label.resize(this->nodes);
  thrust::sequence(h_comity_label.begin(), h_comity_label.end());
  thrust::copy(h_comity_label.begin(), h_comity_label.end(), this->d_comity_label.begin());

  this->d_degree.insert(this->d_degree.begin(), degree.begin(), degree.end());
  this->d_loop.insert(this->d_loop.begin(), loop.begin(), loop.end());
  this->d_finished.insert(this->d_finished.begin(), finished.begin(), finished.end());

  this->d_degree1.resize(this->edges, 0);
  this->d_degree2.resize(this->edges, 0);
  this->d_loop1.resize(this->edges, 0);
  this->d_loop2.resize(this->edges, 0);
  this->d_connect.resize(this->edges, 0);
  this->d_loop_module.resize(this->edges, 0);
  this->d_degree_module.resize(this->edges, 0);
  this->d_entropy_delta.resize(this->edges, 0.0);

  this->d_src_idx.insert(this->d_src_idx.begin(), rows.begin(), rows.end());
  this->d_tgt_idx.insert(this->d_tgt_idx.begin(), cols.begin(), cols.end());
  this->d_weights.insert(this->d_weights.begin(), weights.begin(), weights.end());

  this->h_src_locs.resize(this->nodes);
  this->h_tgt_locs.resize(this->nodes);

  for (int idx = 0; idx < this->nodes; idx++) {
    this->h_src_locs[idx] = thrust::host_vector<int>(src_locs[idx].begin(), src_locs[idx].end());
    this->h_tgt_locs[idx] = thrust::host_vector<int>(tgt_locs[idx].begin(), tgt_locs[idx].end());
  }

  thrust::transform(this->d_degree.begin(), this->d_degree.end(), this->d_loop.begin(), this->d_degree.begin(), thrust::minus<float>());
  thrust::device_vector<float> d_degree_loop(this->nodes);
  thrust::transform(this->d_degree.begin(), this->d_degree.end(), this->d_loop.begin(), d_degree_loop.begin(), thrust::minus<float>());
  this->degree_sum = thrust::reduce(d_degree_loop.begin(), d_degree_loop.end());

  this->d_loop.assign(this->nodes, 0); // 已经删除所有自环，可以去掉loop

  thrust::gather(this->d_src_idx.begin(), this->d_src_idx.end(), this->d_degree.begin(), this->d_degree1.begin());
  thrust::gather(this->d_tgt_idx.begin(), this->d_tgt_idx.end(), this->d_degree.begin(), this->d_degree2.begin());
  thrust::gather(this->d_src_idx.begin(), this->d_src_idx.end(), this->d_loop.begin(), this->d_loop1.begin());
  thrust::gather(this->d_tgt_idx.begin(), this->d_tgt_idx.end(), this->d_loop.begin(), this->d_loop2.begin());

  // connect = weight * 2
  thrust::transform(this->d_weights.begin(), this->d_weights.end(), this->d_connect.begin(), double_functor<float>());

  // degree_module = degree1 + degree2
  thrust::transform(this->d_degree1.begin(), this->d_degree1.end(), this->d_degree2.begin(), this->d_degree_module.begin(), thrust::plus<float>());

  // loop_module = loop1 + loop2 + connect
  thrust::transform(this->d_loop1.begin(), this->d_loop1.end(), this->d_loop2.begin(), this->d_loop_module.begin(), thrust::plus<float>());
  thrust::transform(this->d_loop_module.begin(), this->d_loop_module.end(), this->d_connect.begin(), this->d_loop_module.begin(), thrust::plus<float>());

  thrust::device_vector<float> d_degree_sum_vec(this->d_loop_module.size(), this->degree_sum);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(this->d_degree1.begin(), this->d_loop1.begin(), this->d_degree2.begin(), this->d_loop2.begin(), this->d_degree_module.begin(),
                                                                 this->d_loop_module.begin(), d_degree_sum_vec.begin(), this->d_connect.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(this->d_degree1.end(), this->d_loop1.end(), this->d_degree2.end(), this->d_loop2.end(), this->d_degree_module.end(),
                                                                 this->d_loop_module.end(), d_degree_sum_vec.end(), this->d_connect.end())),
                    this->d_entropy_delta.begin(), entropy_delta_functor<float>());

  thrust::device_vector<int> finish_zero(this->edges, 0);
  thrust::device_vector<int> seq_map(this->edges);
  thrust::sequence(seq_map.begin(), seq_map.end());

  thrust::scatter_if(finish_zero.begin(), finish_zero.end(), seq_map.begin(), this->d_finished.begin(), this->d_entropy_delta.begin(), isFinished());

  this->printf_detail();
}

void HSSInfo::CommunityDetection() {
  //  struct timeval t1, t2, t3;
  //  float timeuse;
  thrust::detail::normal_iterator<thrust::device_ptr<float>> max_index = thrust::max_element(thrust::device, this->d_entropy_delta.begin(), this->d_entropy_delta.end());
  while (*max_index > 0) {
    //    gettimeofday(&t1, nullptr);
    this->Update<int>(max_index - this->d_entropy_delta.begin());
    //    gettimeofday(&t2, nullptr);
    max_index = thrust::max_element(thrust::device, this->d_entropy_delta.begin(), this->d_entropy_delta.end());
    //    gettimeofday(&t3, nullptr);
    //    timeuse = (t2.tv_sec - t1.tv_sec) + (float) (t2.tv_usec - t1.tv_usec) / 1000000.0;
    //    std::cout << std::endl << "update time = " << timeuse << std::endl; //输出时间（单位：ｓ）
    //    timeuse = (t3.tv_sec - t2.tv_sec) + (float) (t3.tv_usec - t2.tv_usec) / 1000000.0;
    //    std::cout << std::endl << "max time = " << timeuse << std::endl; //输出时间（单位：ｓ）
    //    this->printf_detail();
    printf_bar(this->edges, this->bar);
  }
}

template <typename T>
void HSSInfo::Update(const T &update_idx) {
  this->d_entropy_delta[update_idx] = 0;
  this->d_finished[update_idx]      = THRUST_TRUE;
  this->bar += 1;

  T node1 = this->d_src_idx[update_idx];
  T node2 = this->d_tgt_idx[update_idx];

  T comity1 = this->d_comity_label[node1];
  T comity2 = this->d_comity_label[node2];

  //  std::cout << comity1 << " " << comity2 << std::endl;

  this->changed.clear();
  this->changed.shrink_to_fit();

  //当前m1和m2所有的locs位置 (此处需要判断是否已经finished)
  T src_m1_len = this->h_src_locs[comity1].size();
  T tgt_m1_len = this->h_tgt_locs[comity1].size();
  T src_m2_len = this->h_src_locs[comity2].size();
  T tgt_m2_len = this->h_tgt_locs[comity2].size();
  thrust::device_vector<T> d_locs_src_all_m1(this->h_src_locs[comity1].begin(), this->h_src_locs[comity1].end());
  thrust::device_vector<T> d_locs_tgt_all_m1(this->h_tgt_locs[comity1].begin(), this->h_tgt_locs[comity1].end());
  thrust::device_vector<T> d_locs_src_all_m2(this->h_src_locs[comity2].begin(), this->h_src_locs[comity2].end());
  thrust::device_vector<T> d_locs_tgt_all_m2(this->h_tgt_locs[comity2].begin(), this->h_tgt_locs[comity2].end());

  thrust::device_ptr<T> d_locs_src_m1_ptr = this->mallocAsyncThrust<T>(src_m1_len);
  thrust::device_ptr<T> d_locs_tgt_m1_ptr = this->mallocAsyncThrust<T>(tgt_m1_len);
  thrust::device_ptr<T> d_locs_src_m2_ptr = this->mallocAsyncThrust<T>(src_m2_len);
  thrust::device_ptr<T> d_locs_tgt_m2_ptr = this->mallocAsyncThrust<T>(tgt_m2_len);
  src_m1_len                              = this->gather_unfinished<T>(d_locs_src_all_m1.begin(), d_locs_src_all_m1.end(), d_locs_src_m1_ptr);
  tgt_m1_len                              = this->gather_unfinished<T>(d_locs_tgt_all_m1.begin(), d_locs_tgt_all_m1.end(), d_locs_tgt_m1_ptr);
  src_m2_len                              = this->gather_unfinished<T>(d_locs_src_all_m2.begin(), d_locs_src_all_m2.end(), d_locs_src_m2_ptr);
  tgt_m2_len                              = this->gather_unfinished<T>(d_locs_tgt_all_m2.begin(), d_locs_tgt_all_m2.end(), d_locs_tgt_m2_ptr);

  thrust::device_ptr<T> d_idxs_tgt_m1_ptr = this->mallocAsyncThrust<T>(src_m1_len);
  thrust::device_ptr<T> d_idxs_src_m1_ptr = this->mallocAsyncThrust<T>(tgt_m1_len);
  thrust::device_ptr<T> d_idxs_tgt_m2_ptr = this->mallocAsyncThrust<T>(src_m2_len);
  thrust::device_ptr<T> d_idxs_src_m2_ptr = this->mallocAsyncThrust<T>(tgt_m2_len);
  this->gather_idxs(d_locs_src_m1_ptr, d_locs_src_m1_ptr + src_m1_len, 0, d_idxs_tgt_m1_ptr);
  this->gather_idxs(d_locs_tgt_m1_ptr, d_locs_tgt_m1_ptr + tgt_m1_len, 1, d_idxs_src_m1_ptr);
  this->gather_idxs(d_locs_src_m2_ptr, d_locs_src_m2_ptr + src_m2_len, 0, d_idxs_tgt_m2_ptr);
  this->gather_idxs(d_locs_tgt_m2_ptr, d_locs_tgt_m2_ptr + tgt_m2_len, 1, d_idxs_src_m2_ptr);

  thrust::device_ptr<T> d_comity_tgt_m1_ptr = this->mallocAsyncThrust<T>(src_m1_len);
  thrust::device_ptr<T> d_comity_src_m1_ptr = this->mallocAsyncThrust<T>(tgt_m1_len);
  thrust::device_ptr<T> d_comity_tgt_m2_ptr = this->mallocAsyncThrust<T>(src_m2_len);
  thrust::device_ptr<T> d_comity_src_m2_ptr = this->mallocAsyncThrust<T>(tgt_m2_len);
  this->gather_cmty(d_idxs_tgt_m1_ptr, d_idxs_tgt_m1_ptr + src_m1_len, d_comity_tgt_m1_ptr);
  this->gather_cmty(d_idxs_src_m1_ptr, d_idxs_src_m1_ptr + tgt_m1_len, d_comity_src_m1_ptr);
  this->gather_cmty(d_idxs_tgt_m2_ptr, d_idxs_tgt_m2_ptr + src_m2_len, d_comity_tgt_m2_ptr);
  this->gather_cmty(d_idxs_src_m2_ptr, d_idxs_src_m2_ptr + tgt_m2_len, d_comity_src_m2_ptr);

  thrust::sort_by_key(thrust::device, d_comity_tgt_m1_ptr, d_comity_tgt_m1_ptr + src_m1_len, d_locs_src_m1_ptr); //sorted
  thrust::sort_by_key(thrust::device, d_comity_src_m1_ptr, d_comity_src_m1_ptr + tgt_m1_len, d_locs_tgt_m1_ptr); //sorted
  thrust::sort_by_key(thrust::device, d_comity_tgt_m2_ptr, d_comity_tgt_m2_ptr + src_m2_len, d_locs_src_m2_ptr); //sorted
  thrust::sort_by_key(thrust::device, d_comity_src_m2_ptr, d_comity_src_m2_ptr + tgt_m2_len, d_locs_tgt_m2_ptr); //sorted

  thrust::device_ptr<T> d_locs_m1_ptr   = this->mallocAsyncThrust<T>(src_m1_len + tgt_m1_len);
  thrust::device_ptr<T> d_comity_m1_ptr = this->mallocAsyncThrust<T>(src_m1_len + tgt_m1_len);
  thrust::device_ptr<T> d_locs_m2_ptr   = this->mallocAsyncThrust<T>(src_m2_len + tgt_m2_len);
  thrust::device_ptr<T> d_comity_m2_ptr = this->mallocAsyncThrust<T>(src_m2_len + tgt_m2_len);

  T len_m1 = this->merge_cmty_self<T>(d_comity_tgt_m1_ptr, d_comity_tgt_m1_ptr + src_m1_len, d_comity_src_m1_ptr, d_comity_src_m1_ptr + tgt_m1_len, d_locs_src_m1_ptr, d_locs_tgt_m1_ptr,
                                      d_comity_m1_ptr, d_locs_m1_ptr);
  T len_m2 = this->merge_cmty_self<T>(d_comity_tgt_m2_ptr, d_comity_tgt_m2_ptr + src_m2_len, d_comity_src_m2_ptr, d_comity_src_m2_ptr + tgt_m2_len, d_locs_src_m2_ptr, d_locs_tgt_m2_ptr,
                                      d_comity_m2_ptr, d_locs_m2_ptr);
  this->cmty_intersection<T>(d_comity_m1_ptr, d_comity_m1_ptr + len_m1, d_comity_m2_ptr, d_comity_m2_ptr + len_m2, d_locs_m1_ptr, d_locs_m2_ptr);

  T src_locs_comity1_len = src_m1_len + src_m2_len;
  T tgt_locs_comity1_len = tgt_m1_len + tgt_m2_len;

  thrust::device_ptr<T> d_src_locs_comity1_ptr = this->mallocAsyncThrust<T>(src_locs_comity1_len);
  thrust::device_ptr<T> d_tgt_locs_comity1_ptr = this->mallocAsyncThrust<T>(tgt_locs_comity1_len);

  this->merge_cmty<T>(d_comity_tgt_m1_ptr, d_comity_tgt_m1_ptr + src_m1_len, d_comity_tgt_m2_ptr, d_comity_tgt_m2_ptr + src_m2_len, d_locs_src_m1_ptr, d_locs_src_m2_ptr, d_src_locs_comity1_ptr);
  this->merge_cmty<T>(d_comity_src_m1_ptr, d_comity_src_m1_ptr + tgt_m1_len, d_comity_src_m2_ptr, d_comity_src_m2_ptr + tgt_m2_len, d_locs_tgt_m1_ptr, d_locs_tgt_m2_ptr, d_tgt_locs_comity1_ptr);

  this->h_src_locs[comity1].resize(src_locs_comity1_len);
  thrust::copy(d_src_locs_comity1_ptr, d_src_locs_comity1_ptr + src_locs_comity1_len, this->h_src_locs[comity1].begin());
  this->h_src_locs[comity2].clear();
  this->h_src_locs[comity2].shrink_to_fit();

  this->h_tgt_locs[comity1].resize(tgt_locs_comity1_len);
  thrust::copy(d_tgt_locs_comity1_ptr, d_tgt_locs_comity1_ptr + tgt_locs_comity1_len, this->h_tgt_locs[comity1].begin());
  this->h_tgt_locs[comity2].clear();
  this->h_tgt_locs[comity2].shrink_to_fit();

  thrust::device_vector<T> d_comity1_vec(this->h_community[comity1]);
  thrust::device_vector<T> d_comity2_vec(this->h_community[comity2]);

  T d_comity2_vec_len                       = this->h_community[comity2].size();
  thrust::device_ptr<T> d_comity1_label_ptr = this->mallocAsyncThrust<T>(d_comity2_vec_len, comity1);
  thrust::scatter(d_comity1_label_ptr, d_comity1_label_ptr + d_comity2_vec_len, d_comity2_vec.begin(), this->d_comity_label.begin());

  T merge_comity_len                       = this->h_community[comity1].size() + this->h_community[comity2].size();
  thrust::device_ptr<T> d_merge_comity_ptr = this->mallocAsyncThrust<T>(merge_comity_len);
  thrust::merge(thrust::device, d_comity1_vec.begin(), d_comity1_vec.end(), d_comity2_vec.begin(), d_comity2_vec.end(), d_merge_comity_ptr);

  this->h_community[comity1].resize(merge_comity_len);
  thrust::copy(d_merge_comity_ptr, d_merge_comity_ptr + merge_comity_len, this->h_community[comity1].begin());
  this->h_community[comity2].clear();
  this->h_community[comity2].shrink_to_fit();

  this->d_degree[comity1]                   = this->d_degree_module[update_idx];
  this->d_loop[comity1]                     = this->d_loop_module[update_idx];
  thrust::device_ptr<float> d_degree1_merge = this->mallocAsyncThrust<float>(src_locs_comity1_len, this->d_degree_module[update_idx]);
  thrust::device_ptr<float> d_degree2_merge = this->mallocAsyncThrust<float>(tgt_locs_comity1_len, this->d_degree_module[update_idx]);
  thrust::device_ptr<float> d_loop1_merge   = this->mallocAsyncThrust<float>(src_locs_comity1_len, this->d_loop_module[update_idx]);
  thrust::device_ptr<float> d_loop2_merge   = this->mallocAsyncThrust<float>(tgt_locs_comity1_len, this->d_loop_module[update_idx]);

  thrust::scatter(d_degree1_merge, d_degree1_merge + src_locs_comity1_len, d_src_locs_comity1_ptr, this->d_degree1.begin());
  thrust::scatter(d_degree2_merge, d_degree2_merge + tgt_locs_comity1_len, d_tgt_locs_comity1_ptr, this->d_degree2.begin());
  thrust::scatter(d_loop1_merge, d_loop1_merge + src_locs_comity1_len, d_src_locs_comity1_ptr, this->d_loop1.begin());
  thrust::scatter(d_loop2_merge, d_loop2_merge + tgt_locs_comity1_len, d_tgt_locs_comity1_ptr, this->d_loop2.begin());

  T vec_size = this->changed.size();
  if (vec_size > 0) {
    thrust::device_ptr<float> d_degree1_c_ptr       = this->mallocAsyncThrust<float>(vec_size);
    thrust::device_ptr<float> d_degree2_c_ptr       = this->mallocAsyncThrust<float>(vec_size);
    thrust::device_ptr<float> d_loop1_c_ptr         = this->mallocAsyncThrust<float>(vec_size);
    thrust::device_ptr<float> d_loop2_c_ptr         = this->mallocAsyncThrust<float>(vec_size);
    thrust::device_ptr<float> d_degree_module_c_ptr = this->mallocAsyncThrust<float>(vec_size);
    thrust::device_ptr<float> d_loop_module_c_ptr   = this->mallocAsyncThrust<float>(vec_size);
    thrust::device_ptr<float> d_connect_c_ptr       = this->mallocAsyncThrust<float>(vec_size);
    thrust::device_ptr<float> d_degree_sum_c_ptr    = this->mallocAsyncThrust<float>(vec_size, this->degree_sum);
    thrust::device_ptr<float> d_entropy_delta_c_ptr = this->mallocAsyncThrust<float>(vec_size);

    thrust::gather(this->changed.begin(), this->changed.end(), this->d_degree1.begin(), d_degree1_c_ptr);
    thrust::gather(this->changed.begin(), this->changed.end(), this->d_degree2.begin(), d_degree2_c_ptr);
    thrust::gather(this->changed.begin(), this->changed.end(), this->d_loop1.begin(), d_loop1_c_ptr);
    thrust::gather(this->changed.begin(), this->changed.end(), this->d_loop2.begin(), d_loop2_c_ptr);

    thrust::transform(d_degree1_c_ptr, d_degree1_c_ptr + vec_size, d_degree2_c_ptr, d_degree_module_c_ptr, thrust::plus<float>());
    thrust::scatter(d_degree_module_c_ptr, d_degree_module_c_ptr + vec_size, this->changed.begin(), this->d_degree_module.begin());

    thrust::gather(this->changed.begin(), this->changed.end(), this->d_connect.begin(), d_connect_c_ptr);
    thrust::transform(d_loop1_c_ptr, d_loop1_c_ptr + vec_size, d_loop2_c_ptr, d_loop_module_c_ptr, thrust::plus<float>());
    thrust::transform(d_loop_module_c_ptr, d_loop_module_c_ptr + vec_size, d_connect_c_ptr, d_loop_module_c_ptr, thrust::plus<float>());
    thrust::scatter(d_loop_module_c_ptr, d_loop_module_c_ptr + vec_size, this->changed.begin(), this->d_loop_module.begin());

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_degree1_c_ptr, d_loop1_c_ptr, d_degree2_c_ptr, d_loop2_c_ptr, d_degree_module_c_ptr, d_loop_module_c_ptr, d_degree_sum_c_ptr, d_connect_c_ptr)),
        thrust::make_zip_iterator(thrust::make_tuple(d_degree1_c_ptr + vec_size, d_loop1_c_ptr + vec_size, d_degree2_c_ptr + vec_size, d_loop2_c_ptr + vec_size, d_degree_module_c_ptr + vec_size,
                                                     d_loop_module_c_ptr + vec_size, d_degree_sum_c_ptr + vec_size, d_connect_c_ptr + vec_size)),
        d_entropy_delta_c_ptr, entropy_delta_functor<float>());
    thrust::scatter(d_entropy_delta_c_ptr, d_entropy_delta_c_ptr + vec_size, this->changed.begin(), this->d_entropy_delta.begin());

    this->freeAsyncThrust(d_degree1_c_ptr);
    this->freeAsyncThrust(d_degree2_c_ptr);
    this->freeAsyncThrust(d_loop1_c_ptr);
    this->freeAsyncThrust(d_loop2_c_ptr);
    this->freeAsyncThrust(d_degree_module_c_ptr);
    this->freeAsyncThrust(d_loop_module_c_ptr);
    this->freeAsyncThrust(d_connect_c_ptr);
    this->freeAsyncThrust(d_degree_sum_c_ptr);
    this->freeAsyncThrust(d_entropy_delta_c_ptr);
  }

  this->freeAsyncThrust(d_locs_src_m1_ptr);
  this->freeAsyncThrust(d_locs_tgt_m1_ptr);
  this->freeAsyncThrust(d_locs_src_m2_ptr);
  this->freeAsyncThrust(d_locs_tgt_m2_ptr);

  this->freeAsyncThrust(d_idxs_tgt_m1_ptr);
  this->freeAsyncThrust(d_idxs_src_m1_ptr);
  this->freeAsyncThrust(d_idxs_tgt_m2_ptr);
  this->freeAsyncThrust(d_idxs_src_m2_ptr);

  this->freeAsyncThrust(d_comity_tgt_m1_ptr);
  this->freeAsyncThrust(d_comity_src_m1_ptr);
  this->freeAsyncThrust(d_comity_tgt_m2_ptr);
  this->freeAsyncThrust(d_comity_src_m2_ptr);

  this->freeAsyncThrust(d_locs_m1_ptr);
  this->freeAsyncThrust(d_comity_m1_ptr);
  this->freeAsyncThrust(d_locs_m2_ptr);
  this->freeAsyncThrust(d_comity_m2_ptr);

  this->freeAsyncThrust(d_src_locs_comity1_ptr);
  this->freeAsyncThrust(d_tgt_locs_comity1_ptr);

  this->freeAsyncThrust(d_comity1_label_ptr);
  this->freeAsyncThrust(d_merge_comity_ptr);

  this->freeAsyncThrust(d_degree1_merge);
  this->freeAsyncThrust(d_degree2_merge);
  this->freeAsyncThrust(d_loop1_merge);
  this->freeAsyncThrust(d_loop2_merge);

  // cudaStreamSynchronize(stream);
}

template <typename T, typename InputIterator, typename OutputIterator>
T HSSInfo::gather_unfinished(InputIterator first, InputIterator last, OutputIterator result_first) {
  //此处将是否已经完成的结果汇总
  T len                              = last - first;
  thrust::device_ptr<bool> d_fin_ptr = this->mallocAsyncThrust<bool>(len);
  thrust::gather(first, last, this->d_finished.begin(), d_fin_ptr);
  T counts = thrust::count(thrust::device, d_fin_ptr, d_fin_ptr + len, THRUST_FALSE);
  thrust::copy_if(thrust::device, first, last, d_fin_ptr, result_first, isUnfinished());
  this->freeAsyncThrust(d_fin_ptr);
  return counts;
}

template <typename InputIterator, typename OutputIterator>
void HSSInfo::gather_idxs(InputIterator map_first, InputIterator map_last, int type, OutputIterator result_first) {
  if (type == 0) { // src -> tgt
    thrust::gather(map_first, map_last, this->d_tgt_idx.begin(), result_first);
  } else if (type == 1) { // tgt -> src
    thrust::gather(map_first, map_last, this->d_src_idx.begin(), result_first);
  }
}

template <typename InputIterator, typename OutputIterator>
void HSSInfo::gather_cmty(InputIterator map_first, InputIterator map_last, OutputIterator result_first) {
  thrust::gather(map_first, map_last, this->d_comity_label.begin(), result_first);
}

template <typename T, typename InputIterator, typename OutputIterator>
T HSSInfo::merge_cmty_self(InputIterator skey_first, InputIterator skey_end, InputIterator tkey_first, InputIterator tkey_end, InputIterator smap_first, InputIterator tmap_first,
                           OutputIterator result_comity_first, OutputIterator result_locs_first) {
  // 合并模块comity (src and tgt)
  T vec_size = skey_end - skey_first + tkey_end - tkey_first;
  if (vec_size == 0) { return 0; }
  if (skey_end - skey_first == 0) {
    thrust::copy(tkey_first, tkey_end, result_comity_first);
    thrust::copy(tmap_first, tmap_first + vec_size, result_locs_first);
    return tkey_end - tkey_first;
  }
  if (tkey_end - tkey_first == 0) {
    thrust::copy(skey_first, skey_end, result_comity_first);
    thrust::copy(smap_first, smap_first + vec_size, result_locs_first);
    return skey_end - skey_first;
  }

  thrust::device_ptr<T> d_map_ptr           = this->mallocAsyncThrust<T>(vec_size);
  thrust::device_ptr<T> d_cmty_ptr          = this->mallocAsyncThrust<T>(vec_size);
  thrust::device_ptr<bool> repeatItem_ptr   = this->mallocAsyncThrust<bool>(vec_size);
  thrust::device_ptr<bool> finished_vec_ptr = this->mallocAsyncThrust<bool>(vec_size);
  thrust::device_ptr<float> zero_vec_ptr    = this->mallocAsyncThrust<float>(vec_size);
  thrust::fill(repeatItem_ptr, repeatItem_ptr + vec_size, THRUST_FALSE);
  thrust::fill(zero_vec_ptr, zero_vec_ptr + vec_size, 0);

  thrust::merge_by_key(thrust::device, skey_first, skey_end, tkey_first, tkey_end, smap_first, tmap_first, d_cmty_ptr, d_map_ptr);
  // 找出重复的元素，设置为finish元素
  thrust::transform(d_cmty_ptr, d_cmty_ptr + vec_size - 1, d_cmty_ptr + 1, repeatItem_ptr + 1, thrust::equal_to<T>());
  thrust::gather(d_map_ptr, d_map_ptr + vec_size, this->d_finished.begin(), finished_vec_ptr);
  thrust::transform(repeatItem_ptr, repeatItem_ptr + vec_size, finished_vec_ptr, finished_vec_ptr, thrust::logical_or<bool>());
  thrust::scatter(finished_vec_ptr, finished_vec_ptr + vec_size, d_map_ptr, this->d_finished.begin());
  thrust::scatter_if(zero_vec_ptr, zero_vec_ptr + vec_size, d_map_ptr, finished_vec_ptr, this->d_entropy_delta.begin(), isFinished());

  // 将未完成元素复制至新vec返回
  T counts = thrust::count(thrust::device, finished_vec_ptr, finished_vec_ptr + vec_size, THRUST_FALSE);
  this->bar += vec_size - counts;
  thrust::copy_if(thrust::device, d_cmty_ptr, d_cmty_ptr + vec_size, finished_vec_ptr, result_comity_first, isUnfinished());
  thrust::copy_if(thrust::device, d_map_ptr, d_map_ptr + vec_size, finished_vec_ptr, result_locs_first, isUnfinished());

  this->freeAsyncThrust(d_map_ptr);
  this->freeAsyncThrust(d_cmty_ptr);
  this->freeAsyncThrust(repeatItem_ptr);
  this->freeAsyncThrust(finished_vec_ptr);
  this->freeAsyncThrust(zero_vec_ptr);
  return counts;
}

template <typename T, typename InputIterator>
void HSSInfo::cmty_intersection(InputIterator key1_first, InputIterator key1_end, InputIterator key2_first, InputIterator key2_end, InputIterator map1_first, InputIterator map2_first) {
  // 合并模块comity (cmty1 and cmty2)
  T vec_size = key1_end - key1_first + key2_end - key2_first;
  if (vec_size <= 0) { return; }

  thrust::device_ptr<T> d_map_ptr                = this->mallocAsyncThrust<T>(vec_size);
  thrust::device_ptr<T> d_cmty_ptr               = this->mallocAsyncThrust<T>(vec_size);
  thrust::device_ptr<bool> repeatItem_ptr        = this->mallocAsyncThrust<bool>(vec_size);
  thrust::device_ptr<bool> finished_vec_ptr      = this->mallocAsyncThrust<bool>(vec_size);
  thrust::device_ptr<float> zero_vec_ptr         = this->mallocAsyncThrust<float>(vec_size);
  thrust::device_ptr<float> d_connect_first_ptr  = this->mallocAsyncThrust<float>(vec_size - 1);
  thrust::device_ptr<float> d_connect_second_ptr = this->mallocAsyncThrust<float>(vec_size - 1);
  thrust::device_ptr<float> d_connect_plus_ptr   = this->mallocAsyncThrust<float>(vec_size);

  thrust::fill(repeatItem_ptr, repeatItem_ptr + vec_size, THRUST_FALSE);
  thrust::fill(zero_vec_ptr, zero_vec_ptr + vec_size, 0);

  thrust::merge_by_key(thrust::device, key1_first, key1_end, key2_first, key2_end, map1_first, map2_first, d_cmty_ptr, d_map_ptr);

  // 找出重复的元素，设置为finish元素
  thrust::transform(d_cmty_ptr, d_cmty_ptr + vec_size - 1, d_cmty_ptr + 1, repeatItem_ptr + 1, thrust::equal_to<T>());
  thrust::gather(d_map_ptr, d_map_ptr + vec_size, this->d_finished.begin(), finished_vec_ptr);
  thrust::transform(repeatItem_ptr, repeatItem_ptr + vec_size, finished_vec_ptr, finished_vec_ptr, thrust::logical_or<bool>());
  thrust::scatter(finished_vec_ptr, finished_vec_ptr + vec_size, d_map_ptr, this->d_finished.begin());
  thrust::scatter_if(zero_vec_ptr, zero_vec_ptr + vec_size, d_map_ptr, finished_vec_ptr, this->d_entropy_delta.begin(), isFinished());

  thrust::gather(d_map_ptr, d_map_ptr + vec_size - 1, this->d_connect.begin(), d_connect_first_ptr);
  thrust::gather(d_map_ptr + 1, d_map_ptr + vec_size, this->d_connect.begin(), d_connect_second_ptr);
  thrust::gather(d_map_ptr, d_map_ptr + vec_size, this->d_connect.begin(), d_connect_plus_ptr);
  thrust::transform_if(thrust::device, d_connect_first_ptr, d_connect_first_ptr + vec_size - 1, d_connect_second_ptr, finished_vec_ptr + 1, d_connect_plus_ptr, thrust::plus<float>(), isFinished());
  thrust::scatter(d_connect_plus_ptr, d_connect_plus_ptr + vec_size, d_map_ptr, this->d_connect.begin());

  long counts = thrust::count(thrust::device, finished_vec_ptr, finished_vec_ptr + vec_size, THRUST_FALSE);
  this->bar += vec_size - counts;
  this->changed.resize(counts);
  thrust::copy_if(thrust::device, d_map_ptr, d_map_ptr + vec_size, finished_vec_ptr, this->changed.begin(), isUnfinished());

  this->freeAsyncThrust(d_map_ptr);
  this->freeAsyncThrust(d_cmty_ptr);
  this->freeAsyncThrust(repeatItem_ptr);
  this->freeAsyncThrust(finished_vec_ptr);
  this->freeAsyncThrust(zero_vec_ptr);
  this->freeAsyncThrust(d_connect_first_ptr);
  this->freeAsyncThrust(d_connect_second_ptr);
  this->freeAsyncThrust(d_connect_plus_ptr);
}

template <typename T, typename InputIterator, typename OutputIterator>
void HSSInfo::merge_cmty(InputIterator key1_first, InputIterator key1_end, InputIterator key2_first, InputIterator key2_end, InputIterator map1_first, InputIterator map2_first,
                         OutputIterator result_map_first) {
  T vec_size                       = key1_end - key1_first + key2_end - key2_first;
  thrust::device_ptr<T> d_cmty_ptr = this->mallocAsyncThrust<T>(vec_size);
  thrust::merge_by_key(thrust::device, key1_first, key1_end, key2_first, key2_end, map1_first, map2_first, d_cmty_ptr, result_map_first);
  this->freeAsyncThrust(d_cmty_ptr);
}

void HSSInfo::printf_detail() {
#ifdef debug
  printf("degree:\t");
  std::for_each(d_degree.cbegin(), d_degree.cend(), printf_functor_float());
  printf("\n");
  printf("loop:\t");
  std::for_each(d_loop.cbegin(), d_loop.cend(), printf_functor_float());
  printf("\n");

  printf("degree1:\t\t");
  std::for_each(d_degree1.cbegin(), d_degree1.cend(), printf_functor_float());
  printf("\n");
  printf("degree2:\t\t");
  std::for_each(d_degree2.cbegin(), d_degree2.cend(), printf_functor_float());
  printf("\n");
  printf("degree_module:\t");
  std::for_each(d_degree_module.cbegin(), d_degree_module.cend(), printf_functor_float());
  printf("\n");

  printf("loop1:\t\t\t");
  std::for_each(d_loop1.cbegin(), d_loop1.cend(), printf_functor_float());
  printf("\n");
  printf("loop2:\t\t\t");
  std::for_each(d_loop2.cbegin(), d_loop2.cend(), printf_functor_float());
  printf("\n");
  printf("loop_module:\t");
  std::for_each(d_loop_module.cbegin(), d_loop_module.cend(), printf_functor_float());
  printf("\n");

  printf("connect:\t\t");
  std::for_each(d_connect.cbegin(), d_connect.cend(), printf_functor_float());
  printf("\n");

  printf("finished:\t\t");
  std::for_each(d_finished.cbegin(), d_finished.cend(), printf_functor_float());
  printf("\n");

  printf("changed:\t");
  std::for_each(changed.cbegin(), changed.cend(), printf_functor_float());
  printf("\n");

  printf("entropy_delta:\t");
  std::for_each(d_entropy_delta.cbegin(), d_entropy_delta.cend(), printf_functor_float());
  printf("\n");

  for (int i = 0; i < this->h_community.size(); i++) {
    if (!this->h_community[i].empty()) { this->printf_detail(" ", this->h_community[i].cbegin(), this->h_community[i].cend()); }
  }

#endif
}
template <typename InputIterator>
void HSSInfo::printf_detail(std::string name, InputIterator first, InputIterator last) {
#ifdef debug
  std::cout << name << ":\t";
  std::for_each(first, last, printf_functor_float());
  printf("\n");
#endif
}
template <typename T>
void HSSInfo::printf_bar(T edgeNum, T now) {
#ifdef debug
  float perc = now * 100.0 / (edgeNum - 1);
  printf("\rProcessing => [%.2f%%]  %d / %d:", perc, now, edgeNum);
  //  for (int j = 1; j <= perc; j++) { printf("█"); }
  fflush(stdout);
#endif
}

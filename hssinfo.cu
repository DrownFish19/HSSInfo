#include "hssinfo.hpp"

#include <sys/time.h>
#include <thrust/count.h>
#include <thrust/device_delete.h>
#include <thrust/device_free.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/merge.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <fstream>
//#define debug
HSSInfo::HSSInfo(const int &nodes, const std::string &filename) : nodes(nodes) {
  std::vector<int> rows;
  std::vector<int> cols;
  std::vector<int> weights;
  std::vector<std::vector<int>> src_locs;
  std::vector<std::vector<int>> tgt_locs;
  src_locs.resize(nodes);
  tgt_locs.resize(nodes);

  std::vector<int> degree;
  std::vector<int> loop;
  std::vector<bool> finished;
  degree.resize(nodes, 0);
  loop.resize(nodes, 0);

  std::cout << "reading " << filename << std::endl;
  std::ifstream in(filename);
  std::string line;
  int count = 0;
  while (getline(in, line)) {   //将in文件中的每一行字符读入到string line中
    std::stringstream ss(line); //使用string初始化stringstream
    int row;
    int col;
    //    int weight;
    ss >> row;
    ss >> col;
    //    ss >> weight;
    rows.push_back(row);
    cols.push_back(col);
    weights.push_back(1);

    degree[row] += 1; //is weight
    degree[col] += 1; //is weight
    if (row == col) {
      loop[row] += 1; // is weight
      finished.push_back(true);
    } else {
      finished.push_back(false);
    }

    src_locs[row].emplace_back(count);
    tgt_locs[col].emplace_back(count);

    count++;
    this->printf_bar(count, count - 1);
  }
  in.close();
  std::cout << std::endl << "read finish" << std::endl;

  this->edges = count;

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

  this->h_degree1.resize(this->edges, 0);
  this->h_degree2.resize(this->edges, 0);
  this->h_loop1.resize(this->edges, 0);
  this->h_loop2.resize(this->edges, 0);

  this->d_src_idx.insert(this->d_src_idx.begin(), rows.begin(), rows.end());
  this->d_tgt_idx.insert(this->d_tgt_idx.begin(), cols.begin(), cols.end());
  this->d_weights.insert(this->d_weights.begin(), weights.begin(), weights.end());

  this->h_src_locs.resize(this->nodes);
  this->h_tgt_locs.resize(this->nodes);
  for (int idx = 0; idx < this->nodes; idx++) {
    this->h_src_locs[idx] = thrust::host_vector<int>(src_locs[idx].begin(), src_locs[idx].end());
    this->h_tgt_locs[idx] = thrust::host_vector<int>(tgt_locs[idx].begin(), tgt_locs[idx].end());
  }

  thrust::transform(this->d_degree.begin(), this->d_degree.end(), this->d_loop.begin(), this->d_degree.begin(), thrust::minus<int>());
  thrust::device_vector<int> d_degree_loop(this->nodes);
  thrust::transform(this->d_degree.begin(), this->d_degree.end(), this->d_loop.begin(), d_degree_loop.begin(), thrust::minus<int>());
  this->degree_sum = thrust::reduce(d_degree_loop.begin(), d_degree_loop.end());

  this->d_loop.assign(this->nodes, 0); // 已经删除所有自环，可以去掉loop

  thrust::gather(this->d_src_idx.begin(), this->d_src_idx.end(), this->d_degree.begin(), this->d_degree1.begin());
  thrust::gather(this->d_tgt_idx.begin(), this->d_tgt_idx.end(), this->d_degree.begin(), this->d_degree2.begin());
  thrust::gather(this->d_src_idx.begin(), this->d_src_idx.end(), this->d_loop.begin(), this->d_loop1.begin());
  thrust::gather(this->d_tgt_idx.begin(), this->d_tgt_idx.end(), this->d_loop.begin(), this->d_loop2.begin());

  // connect = weight * 2
  thrust::transform(this->d_weights.begin(), this->d_weights.end(), this->d_connect.begin(), double_functor<int>());

  // degree_module = degree1 + degree2
  thrust::transform(this->d_degree1.begin(), this->d_degree1.end(), this->d_degree2.begin(), this->d_degree_module.begin(), thrust::plus<int>());

  // loop_module = loop1 + loop2 + connect
  thrust::transform(this->d_loop1.begin(), this->d_loop1.end(), this->d_loop2.begin(), this->d_loop_module.begin(), thrust::plus<int>());
  thrust::transform(this->d_loop_module.begin(), this->d_loop_module.end(), this->d_connect.begin(), this->d_loop_module.begin(), thrust::plus<int>());

  thrust::device_vector<int> d_degree_sum_vec(this->d_loop_module.size(), this->degree_sum);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(this->d_degree1.begin(), this->d_loop1.begin(), this->d_degree2.begin(), this->d_loop2.begin(),
                                                                 this->d_degree_module.begin(), this->d_loop_module.begin(), d_degree_sum_vec.begin(), this->d_connect.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(this->d_degree1.end(), this->d_loop1.end(), this->d_degree2.end(), this->d_loop2.end(),
                                                                 this->d_degree_module.end(), this->d_loop_module.end(), d_degree_sum_vec.end(), this->d_connect.end())),
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

  int node1 = this->d_src_idx[update_idx];
  int node2 = this->d_tgt_idx[update_idx];

  int comity1 = this->d_comity_label[node1];
  int comity2 = this->d_comity_label[node2];

  //  std::cout << comity1 << " " << comity2 << std::endl;

  this->changed.clear();
  this->changed.shrink_to_fit();

  //当前m1和m2所有的locs位置 (此处需要判断是否已经finished)
  thrust::device_vector<T> d_locs_src_all_m1(this->h_src_locs[comity1].begin(), this->h_src_locs[comity1].end());
  thrust::device_vector<T> d_locs_tgt_all_m1(this->h_tgt_locs[comity1].begin(), this->h_tgt_locs[comity1].end());
  thrust::device_vector<T> d_locs_src_all_m2(this->h_src_locs[comity2].begin(), this->h_src_locs[comity2].end());
  thrust::device_vector<T> d_locs_tgt_all_m2(this->h_tgt_locs[comity2].begin(), this->h_tgt_locs[comity2].end());

  thrust::device_vector<T> d_locs_src_m1(d_locs_src_all_m1.size());
  thrust::device_vector<T> d_locs_tgt_m1(d_locs_tgt_all_m1.size());
  thrust::device_vector<T> d_locs_src_m2(d_locs_src_all_m2.size());
  thrust::device_vector<T> d_locs_tgt_m2(d_locs_tgt_all_m2.size());
  T src_m1_len = this->gather_unfinished<T>(d_locs_src_all_m1.begin(), d_locs_src_all_m1.end(), d_locs_src_m1.begin());
  T tgt_m1_len = this->gather_unfinished<T>(d_locs_tgt_all_m1.begin(), d_locs_tgt_all_m1.end(), d_locs_tgt_m1.begin());
  T src_m2_len = this->gather_unfinished<T>(d_locs_src_all_m2.begin(), d_locs_src_all_m2.end(), d_locs_src_m2.begin());
  T tgt_m2_len = this->gather_unfinished<T>(d_locs_tgt_all_m2.begin(), d_locs_tgt_all_m2.end(), d_locs_tgt_m2.begin());

  thrust::device_vector<T> d_idxs_tgt_m1(src_m1_len);
  thrust::device_vector<T> d_idxs_src_m1(tgt_m1_len);
  thrust::device_vector<T> d_idxs_tgt_m2(src_m2_len);
  thrust::device_vector<T> d_idxs_src_m2(tgt_m2_len);
  this->gather_idxs<T>(d_locs_src_m1.begin(), d_locs_src_m1.begin() + src_m1_len, 0, d_idxs_tgt_m1.begin());
  this->gather_idxs<T>(d_locs_tgt_m1.begin(), d_locs_tgt_m1.begin() + tgt_m1_len, 1, d_idxs_src_m1.begin());
  this->gather_idxs<T>(d_locs_src_m2.begin(), d_locs_src_m2.begin() + src_m2_len, 0, d_idxs_tgt_m2.begin());
  this->gather_idxs<T>(d_locs_tgt_m2.begin(), d_locs_tgt_m2.begin() + tgt_m2_len, 1, d_idxs_src_m2.begin());

  thrust::device_vector<T> d_comity_tgt_m1(src_m1_len);
  thrust::device_vector<T> d_comity_src_m1(tgt_m1_len);
  thrust::device_vector<T> d_comity_tgt_m2(src_m2_len);
  thrust::device_vector<T> d_comity_src_m2(tgt_m2_len);
  this->gather_cmty<T>(d_idxs_tgt_m1.begin(), d_idxs_tgt_m1.end(), d_comity_tgt_m1.begin());
  this->gather_cmty<T>(d_idxs_src_m1.begin(), d_idxs_src_m1.end(), d_comity_src_m1.begin());
  this->gather_cmty<T>(d_idxs_tgt_m2.begin(), d_idxs_tgt_m2.end(), d_comity_tgt_m2.begin());
  this->gather_cmty<T>(d_idxs_src_m2.begin(), d_idxs_src_m2.end(), d_comity_src_m2.begin());

  thrust::sort_by_key(thrust::device, d_comity_tgt_m1.begin(), d_comity_tgt_m1.end(), d_locs_src_m1.begin()); //sorted
  thrust::sort_by_key(thrust::device, d_comity_src_m1.begin(), d_comity_src_m1.end(), d_locs_tgt_m1.begin()); //sorted
  thrust::sort_by_key(thrust::device, d_comity_tgt_m2.begin(), d_comity_tgt_m2.end(), d_locs_src_m2.begin()); //sorted
  thrust::sort_by_key(thrust::device, d_comity_src_m2.begin(), d_comity_src_m2.end(), d_locs_tgt_m2.begin()); //sorted

  thrust::device_vector<T> d_locs_m1(d_comity_tgt_m1.size() + d_comity_src_m1.size());
  thrust::device_vector<T> d_comity_m1(d_comity_tgt_m1.size() + d_comity_src_m1.size());

  T len_m1 = this->merge_cmty_self<T>(d_comity_tgt_m1.begin(), d_comity_tgt_m1.end(), d_comity_src_m1.begin(), d_comity_src_m1.end(), d_locs_src_m1.begin(), d_locs_tgt_m1.begin(),
                                      d_comity_m1.begin(), d_locs_m1.begin());

  thrust::device_vector<T> d_locs_m2(d_comity_tgt_m2.size() + d_comity_src_m2.size());
  thrust::device_vector<T> d_comity_m2(d_comity_tgt_m2.size() + d_comity_src_m2.size());
  T len_m2 = this->merge_cmty_self<T>(d_comity_tgt_m2.begin(), d_comity_tgt_m2.end(), d_comity_src_m2.begin(), d_comity_src_m2.end(), d_locs_src_m2.begin(), d_locs_tgt_m2.begin(),
                                      d_comity_m2.begin(), d_locs_m2.begin());

  this->cmty_intersection<T>(d_comity_m1.begin(), d_comity_m1.begin() + len_m1, d_comity_m2.begin(), d_comity_m2.begin() + len_m2, d_locs_m1.begin(), d_locs_m2.begin());

  thrust::device_vector<T> d_src_locs_comity1(d_comity_tgt_m1.size() + d_comity_tgt_m2.size());
  this->merge_cmty<T>(d_comity_tgt_m1.begin(), d_comity_tgt_m1.end(), d_comity_tgt_m2.begin(), d_comity_tgt_m2.end(), d_locs_src_m1.begin(), d_locs_src_m2.begin(),
                      d_src_locs_comity1.begin());
  this->h_src_locs[comity1].resize(d_src_locs_comity1.size());
  thrust::copy(d_src_locs_comity1.begin(), d_src_locs_comity1.end(), this->h_src_locs[comity1].begin());
  this->h_src_locs[comity2].clear();
  this->h_src_locs[comity2].shrink_to_fit();

  thrust::device_vector<T> d_tgt_locs_comity1(d_comity_src_m1.size() + d_comity_src_m2.size());
  this->merge_cmty<T>(d_comity_src_m1.begin(), d_comity_src_m1.end(), d_comity_src_m2.begin(), d_comity_src_m2.end(), d_locs_tgt_m1.begin(), d_locs_tgt_m2.begin(),
                      d_tgt_locs_comity1.begin());
  this->h_tgt_locs[comity1].resize(d_tgt_locs_comity1.size());
  thrust::copy(d_tgt_locs_comity1.begin(), d_tgt_locs_comity1.end(), this->h_tgt_locs[comity1].begin());
  this->h_tgt_locs[comity2].clear();
  this->h_tgt_locs[comity2].shrink_to_fit();

  thrust::device_vector<T> d_comity1_vec(this->h_community[comity1]);
  thrust::device_vector<T> d_comity2_vec(this->h_community[comity2]);
  thrust::device_vector<T> d_comity1_label(d_comity2_vec.size(), comity1);
  thrust::scatter(d_comity1_label.begin(), d_comity1_label.end(), d_comity2_vec.begin(), this->d_comity_label.begin());

  thrust::device_vector<T> d_merge_comity(this->h_community[comity1].size() + this->h_community[comity2].size());
  thrust::merge(thrust::device, d_comity1_vec.begin(), d_comity1_vec.end(), d_comity2_vec.begin(), d_comity2_vec.end(), d_merge_comity.begin());
  this->h_community[comity1].resize(d_merge_comity.size());
  thrust::copy(d_merge_comity.begin(), d_merge_comity.end(), this->h_community[comity1].begin());
  this->h_community[comity2].clear();
  this->h_community[comity2].shrink_to_fit();

  this->d_degree[comity1] = this->d_degree_module[update_idx];
  this->d_loop[comity1]   = this->d_loop_module[update_idx];

  thrust::device_vector<int> d_degree1_merge(d_src_locs_comity1.size(), this->d_degree_module[update_idx]);
  thrust::scatter(d_degree1_merge.begin(), d_degree1_merge.end(), d_src_locs_comity1.begin(), this->d_degree1.begin());

  thrust::device_vector<int> d_degree2_merge(d_tgt_locs_comity1.size(), this->d_degree_module[update_idx]);
  thrust::scatter(d_degree2_merge.begin(), d_degree2_merge.end(), d_tgt_locs_comity1.begin(), this->d_degree2.begin());

  thrust::device_vector<int> d_loop1_merge(d_src_locs_comity1.size(), this->d_loop_module[update_idx]);
  thrust::scatter(d_loop1_merge.begin(), d_loop1_merge.end(), d_src_locs_comity1.begin(), this->d_loop1.begin());

  thrust::device_vector<int> d_loop2_merge(d_tgt_locs_comity1.size(), this->d_loop_module[update_idx]);
  thrust::scatter(d_loop2_merge.begin(), d_loop2_merge.end(), d_tgt_locs_comity1.begin(), this->d_loop2.begin());

  unsigned long vec_size = this->changed.size();
  if (vec_size > 0) {
    thrust::device_vector<T> d_degree1_c(vec_size);
    thrust::device_vector<T> d_degree2_c(vec_size);
    thrust::device_vector<T> d_loop1_c(vec_size);
    thrust::device_vector<T> d_loop2_c(vec_size);
    thrust::device_vector<T> d_degree_module_c(vec_size);
    thrust::device_vector<T> d_loop_module_c(vec_size);
    thrust::device_vector<T> d_connect_c(vec_size);
    thrust::device_vector<T> d_degree_sum_c(vec_size, this->degree_sum);
    thrust::device_vector<float> d_entropy_delta_c(vec_size);

    thrust::gather(this->changed.begin(), this->changed.end(), this->d_degree1.begin(), d_degree1_c.begin());
    thrust::gather(this->changed.begin(), this->changed.end(), this->d_degree2.begin(), d_degree2_c.begin());
    thrust::gather(this->changed.begin(), this->changed.end(), this->d_loop1.begin(), d_loop1_c.begin());
    thrust::gather(this->changed.begin(), this->changed.end(), this->d_loop2.begin(), d_loop2_c.begin());

    thrust::transform(d_degree1_c.begin(), d_degree1_c.end(), d_degree2_c.begin(), d_degree_module_c.begin(), thrust::plus<T>());
    thrust::scatter(d_degree_module_c.begin(), d_degree_module_c.end(), this->changed.begin(), this->d_degree_module.begin());

    thrust::gather(this->changed.begin(), this->changed.end(), this->d_connect.begin(), d_connect_c.begin());
    thrust::transform(d_loop1_c.begin(), d_loop1_c.end(), d_loop2_c.begin(), d_loop_module_c.begin(), thrust::plus<T>());
    thrust::transform(d_loop_module_c.begin(), d_loop_module_c.end(), d_connect_c.begin(), d_loop_module_c.begin(), thrust::plus<T>());
    thrust::scatter(d_loop_module_c.begin(), d_loop_module_c.end(), this->changed.begin(), this->d_loop_module.begin());

    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_degree1_c.begin(), d_loop1_c.begin(), d_degree2_c.begin(), d_loop2_c.begin(), d_degree_module_c.begin(),
                                                                   d_loop_module_c.begin(), d_degree_sum_c.begin(), d_connect_c.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(d_degree1_c.end(), d_loop1_c.end(), d_degree2_c.end(), d_loop2_c.end(), d_degree_module_c.end(),
                                                                   d_loop_module_c.end(), d_degree_sum_c.end(), d_connect_c.end())),
                      d_entropy_delta_c.begin(), entropy_delta_functor<float>());
    thrust::scatter(d_entropy_delta_c.begin(), d_entropy_delta_c.end(), this->changed.begin(), this->d_entropy_delta.begin());

#ifdef debug
    printf_detail("d_degree1_c", d_degree1_c.cbegin(), d_degree1_c.cend());
    printf_detail("d_degree2_c", d_degree2_c.cbegin(), d_degree2_c.cend());
    printf_detail("d_loop1_c", d_loop1_c.cbegin(), d_loop1_c.cend());
    printf_detail("d_loop2_c", d_loop2_c.cbegin(), d_loop2_c.cend());
    printf_detail("d_degree_module_c", d_degree_module_c.cbegin(), d_degree_module_c.cend());
    printf_detail("d_loop_module_c", d_loop_module_c.cbegin(), d_loop_module_c.cend());
    printf_detail("d_connect_c", d_connect_c.cbegin(), d_connect_c.cend());
    printf_detail("d_degree_sum_c", d_degree_sum_c.cbegin(), d_degree_sum_c.cend());
    printf_detail("d_entropy_delta_c", d_entropy_delta_c.cbegin(), d_entropy_delta_c.cend());
#endif
  }
}

template <typename T, typename InputIterator, typename OutputIterator>
T HSSInfo::gather_unfinished(InputIterator first, InputIterator last, OutputIterator result_first) {
  //此处将是否已经完成的结果汇总
  thrust::device_vector<bool> d_fin_vec(last - first);
  thrust::gather(first, last, this->d_finished.begin(), d_fin_vec.begin());
  T counts = thrust::count(thrust::device, d_fin_vec.begin(), d_fin_vec.end(), THRUST_FALSE);
  thrust::copy_if(thrust::device, first, last, d_fin_vec.begin(), result_first, isUnfinished());
  return counts;
}

template <typename T, typename InputIterator, typename OutputIterator>
void HSSInfo::gather_idxs(InputIterator map_first, InputIterator map_last, int type, OutputIterator result_first) {
  if (type == 0) { // src -> tgt
    thrust::gather(map_first, map_last, this->d_tgt_idx.begin(), result_first);
  } else if (type == 1) { // tgt -> src
    thrust::gather(map_first, map_last, this->d_src_idx.begin(), result_first);
  }
}

template <typename T, typename InputIterator, typename OutputIterator>
void HSSInfo::gather_cmty(InputIterator map_first, InputIterator map_last, OutputIterator result_first) {
  thrust::gather(map_first, map_last, this->d_comity_label.begin(), result_first);
}

template <typename T, typename InputIterator, typename OutputIterator>
T HSSInfo::merge_cmty_self(InputIterator skey_first, InputIterator skey_end, InputIterator tkey_first, InputIterator tkey_end, InputIterator smap_first,
                                                      InputIterator tmap_first, OutputIterator result_comity_first, OutputIterator result_locs_first) {
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

  thrust::device_vector<T> d_map(vec_size);
  thrust::device_vector<T> d_cmty(vec_size);
  thrust::merge_by_key(thrust::device, skey_first, skey_end, tkey_first, tkey_end, smap_first, tmap_first, d_cmty.begin(), d_map.begin());

  // 找出重复的元素，设置为finish元素
  thrust::device_vector<bool> repeatItem(vec_size, THRUST_FALSE);
  thrust::transform(d_cmty.begin(), d_cmty.end() - 1, d_cmty.begin() + 1, repeatItem.begin() + 1, thrust::equal_to<T>());
  thrust::device_vector<bool> finished_vec(vec_size);
  thrust::gather(d_map.begin(), d_map.end(), this->d_finished.begin(), finished_vec.begin());
  thrust::transform(repeatItem.begin(), repeatItem.end(), finished_vec.begin(), finished_vec.begin(), thrust::logical_or<bool>());
  thrust::scatter(finished_vec.begin(), finished_vec.end(), d_map.begin(), this->d_finished.begin());
  thrust::device_vector<bool> zero_vec(vec_size, 0);
  thrust::scatter_if(zero_vec.begin(), zero_vec.end(), d_map.begin(), finished_vec.begin(), this->d_entropy_delta.begin(), isFinished());

  // 将未完成元素复制至新vec返回
  T counts = thrust::count(thrust::device, finished_vec.begin(), finished_vec.end(), THRUST_FALSE);
  this->bar += finished_vec.size() - counts;
  thrust::copy_if(thrust::device, d_cmty.begin(), d_cmty.end(), finished_vec.begin(), result_comity_first, isUnfinished());
  thrust::copy_if(thrust::device, d_map.begin(), d_map.end(), finished_vec.begin(), result_locs_first, isUnfinished());
  return counts;
}

template <typename T, typename InputIterator>
void HSSInfo::cmty_intersection(InputIterator key1_first, InputIterator key1_end, InputIterator key2_first, InputIterator key2_end,
                                                           InputIterator map1_first, InputIterator map2_first) {
  // 合并模块comity (cmty1 and cmty2)
  T vec_size = key1_end - key1_first + key2_end - key2_first;
  if (vec_size <= 0) { return; }

  thrust::device_vector<T> d_map(vec_size);
  thrust::device_vector<T> d_cmty(vec_size);
  thrust::merge_by_key(thrust::device, key1_first, key1_end, key2_first, key2_end, map1_first, map2_first, d_cmty.begin(), d_map.begin());

  // 找出重复的元素，设置为finish元素
  thrust::device_vector<bool> repeatItem(vec_size, THRUST_FALSE);
  thrust::transform(d_cmty.begin(), d_cmty.end() - 1, d_cmty.begin() + 1, repeatItem.begin() + 1, thrust::equal_to<T>());
  thrust::device_vector<bool> finished_vec(vec_size);
  thrust::gather(d_map.begin(), d_map.end(), this->d_finished.begin(), finished_vec.begin());
  thrust::transform(repeatItem.begin(), repeatItem.end(), finished_vec.begin(), finished_vec.begin(), thrust::logical_or<bool>());
  thrust::scatter(finished_vec.begin(), finished_vec.end(), d_map.begin(), this->d_finished.begin());
  thrust::device_vector<T> zero_vec(vec_size, 0);
  thrust::scatter_if(zero_vec.begin(), zero_vec.end(), d_map.begin(), finished_vec.begin(), this->d_entropy_delta.begin(), isFinished());

  thrust::device_vector<T> d_connect_first(vec_size - 1);
  thrust::device_vector<T> d_connect_second(vec_size - 1);
  thrust::device_vector<T> d_connect_plus(vec_size);
  thrust::gather(d_map.begin(), d_map.end() - 1, this->d_connect.begin(), d_connect_first.begin());
  thrust::gather(d_map.begin() + 1, d_map.end(), this->d_connect.begin(), d_connect_second.begin());
  thrust::gather(d_map.begin(), d_map.end(), this->d_connect.begin(), d_connect_plus.begin());

  thrust::transform_if(thrust::device, d_connect_first.begin(), d_connect_first.end(), d_connect_second.begin(), finished_vec.begin() + 1, d_connect_plus.begin(),
                       thrust::plus<T>(), isFinished());
  thrust::scatter(d_connect_plus.begin(), d_connect_plus.end(), d_map.begin(), this->d_connect.begin());

  long counts = thrust::count(thrust::device, finished_vec.begin(), finished_vec.end(), THRUST_FALSE);
  this->bar += finished_vec.size() - counts;
  this->changed.resize(counts);
  thrust::copy_if(thrust::device, d_map.begin(), d_map.end(), finished_vec.begin(), this->changed.begin(), isUnfinished());
}

template <typename T, typename InputIterator, typename OutputIterator>
void HSSInfo::merge_cmty(InputIterator key1_first, InputIterator key1_end, InputIterator key2_first, InputIterator key2_end, InputIterator map1_first,
                                                    InputIterator map2_first, OutputIterator result_map_first) {
  T vec_size = key1_end - key1_first + key2_end - key2_first;
  thrust::device_vector<T> d_cmty(vec_size);
  thrust::merge_by_key(thrust::device, key1_first, key1_end, key2_first, key2_end, map1_first, map2_first, d_cmty.begin(), result_map_first);
}

void HSSInfo::printf_detail() {
#ifdef debug
  printf("degree:\t");
  std::for_each(d_degree.cbegin(), d_degree.cend(), printf_functor_int());
  printf("\n");
  printf("loop:\t");
  std::for_each(d_loop.cbegin(), d_loop.cend(), printf_functor_int());
  printf("\n");

  printf("degree1:\t\t");
  std::for_each(d_degree1.cbegin(), d_degree1.cend(), printf_functor_int());
  printf("\n");
  printf("degree2:\t\t");
  std::for_each(d_degree2.cbegin(), d_degree2.cend(), printf_functor_int());
  printf("\n");
  printf("degree_module:\t");
  std::for_each(d_degree_module.cbegin(), d_degree_module.cend(), printf_functor_int());
  printf("\n");

  printf("loop1:\t\t\t");
  std::for_each(d_loop1.cbegin(), d_loop1.cend(), printf_functor_int());
  printf("\n");
  printf("loop2:\t\t\t");
  std::for_each(d_loop2.cbegin(), d_loop2.cend(), printf_functor_int());
  printf("\n");
  printf("loop_module:\t");
  std::for_each(d_loop_module.cbegin(), d_loop_module.cend(), printf_functor_int());
  printf("\n");

  printf("connect:\t\t");
  std::for_each(d_connect.cbegin(), d_connect.cend(), printf_functor_int());
  printf("\n");

  printf("finished:\t\t");
  std::for_each(d_finished.cbegin(), d_finished.cend(), printf_functor_int());
  printf("\n");

  printf("changed:\t");
  std::for_each(changed.cbegin(), changed.cend(), printf_functor_int());
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
  float perc = now * 100.0 / (edgeNum - 1);
  printf("\rProcessing => [%.2f%%]  %d / %d:", perc, now, edgeNum);
  //  for (int j = 1; j <= perc; j++) { printf("█"); }
  fflush(stdout);
}
void HSSInfo::output_clusters(const std::string &filename) {
  std::ofstream fout;
  fout.open(filename, std::ios::out | std::ios::ate);

  for (auto i = 0; i < this->h_community.size(); i++) {
    if (!this->h_community[i].empty()) {
      for (auto index = this->h_community[i].cbegin(); index < this->h_community[i].cend(); index++) { fout << *index << " "; }
      fout << std::endl;
    }
  }
}

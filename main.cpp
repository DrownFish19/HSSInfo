#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <ostream>

#include "hssinfo.hpp"

int main(int argc, char* argv[]) {
  //  int nodes     = 6;
  //  int edges     = 9;
  //  int rows[]    = {0, 0, 1, 1, 2, 3, 3, 4, 5};
  //  int cols[]    = {1, 4, 2, 4, 4, 3, 4, 5, 5};
  //  int weights[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  //  int nodes     = 20;
  //  int edges     = 33;
  //  int rows[]    = {0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 7, 7, 9, 10, 10, 11, 11, 11, 12, 12, 15, 15, 16, 16};
  //  int cols[]    = {15, 17, 16, 17, 7, 8, 15, 11, 13, 15, 9, 13, 14, 19, 13, 15, 17, 19, 18, 12, 14, 11, 12, 17, 16, 17, 19, 13, 15, 16, 19, 17, 19};
  //  int weights[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  //  thrust::host_vector<int> h_rows(rows, rows + edges);
  //  thrust::host_vector<int> h_cols(cols, cols + edges);
  //  thrust::host_vector<int> h_weights(weights, weights + edges);

  std::string filename;
  std::string filename_output;
  int nodes;

  int opt = 0;
  bool flag_i, flag_o, flag_n = false;
  while ((opt = getopt(argc, argv, "hi:o:n:")) != -1) {
    switch (opt) {
      case 'h': printf("Usage: ./HSSInfo -i <InputFile> -o <OutFile> -n <NodeNum>\n"); return 0;
      case 'i':
        filename = std::string(optarg);
        flag_i   = true;
        break;
      case 'o':
        filename_output = std::string(optarg);
        flag_o          = true;
        break;
      case 'n':
        nodes  = int(strtoul(optarg, nullptr, 10));
        flag_n = true;
        break;
    }
  }
  if (flag_i & flag_o & flag_n) {
    printf("input_file = %s, output_file = %s, process_num = %d\n\n", filename.c_str(), filename_output.c_str(), nodes);
  } else {
    printf("Usage: ./HSSInfo -i <InputFile> -o <OutFile> -n <NodeNum>\n");
    return 0;
  }

  struct timeval t1, t2, t3;
  double timeuse;
  gettimeofday(&t1, nullptr);

  std::vector<int> rows, cols;
  std::vector<float> weights;
  std::ifstream in(filename);
  std::string line;
  while (getline(in, line)) {   //将in文件中的每一行字符读入到string line中
    std::stringstream ss(line); //使用string初始化stringstream
    int row;
    int col;
    float weight = 1.0;
    ss >> row;
    ss >> col;
    ss >> weight;
    rows.push_back(row);
    cols.push_back(col);
    weights.push_back(weight);
  }

  HSSInfo info(nodes, rows, cols, weights);
  info.output_edge_info();
  gettimeofday(&t2, nullptr);
  timeuse = (float) (t2.tv_sec - t1.tv_sec) + (float) (t2.tv_usec - t1.tv_usec) / 1000000.0;
  std::cout << "init time = " << timeuse << std::endl; //输出时间（单位：ｓ）

  std::cout << std::endl << "community detection start" << std::endl;
  info.CommunityDetection();

  gettimeofday(&t3, nullptr);
  timeuse = (float) (t3.tv_sec - t2.tv_sec) + (float) (t3.tv_usec - t2.tv_usec) / 1000000.0;
  std::cout << std::endl << "community detection time = " << timeuse << std::endl; //输出时间（单位：ｓ）

  std::ofstream fout;
  fout.open(filename_output, std::ios::out | std::ios::ate);
  for (auto i = 0; i < info.h_community.size(); i++) {
    if (!info.h_community[i].empty()) {
      for (auto index = info.h_community[i].cbegin(); index < info.h_community[i].cend(); index++) { fout << *index << " "; }
      fout << std::endl;
    }
  }

  info.output_node_info();
}

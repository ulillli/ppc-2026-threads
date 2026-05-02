#include "papulina_y_radix_sort/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

namespace papulina_y_radix_sort {

PapulinaYRadixSortSTL::PapulinaYRadixSortSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool PapulinaYRadixSortSTL::ValidationImpl() {
  return !GetInput().empty();
}

bool PapulinaYRadixSortSTL::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool PapulinaYRadixSortSTL::RunImpl() {
  size_t size = GetInput().size();
  if (size == 0) return true;

  std::vector<double> data = GetInput();
  RadixSortParallel(data.data(), size);

  GetOutput() = std::move(data);
  return true;
}

bool PapulinaYRadixSortSTL::PostProcessingImpl() {
  return true;
}

uint64_t PapulinaYRadixSortSTL::InBytes(double d) {
  const uint64_t kMask = 0x8000000000000000;
  uint64_t bits = 0;
  memcpy(&bits, &d, sizeof(double));
  if ((bits & kMask) != 0) {
    bits = ~bits;
  } else {
    bits = bits ^ kMask;
  }
  return bits;
}

double PapulinaYRadixSortSTL::FromBytes(uint64_t bits) {
  const uint64_t kMask = 0x8000000000000000;
  double d = 0;
  if ((bits & kMask) != 0) {
    bits = bits ^ kMask;
  } else {
    bits = ~bits;
  }
  memcpy(&d, &bits, sizeof(double));
  return d;
}

void PapulinaYRadixSortSTL::RadixSortParallel(double *arr, size_t size) {
  int num_threads = std::thread::hardware_concurrency();
  if (num_threads < 1) num_threads = 1;
  if (size < 1000) num_threads = 1;

  std::vector<uint64_t> bytes(size);
  std::vector<uint64_t> temp(size);

  for (size_t i = 0; i < size; ++i) {
    bytes[i] = InBytes(arr[i]);
  }

  uint64_t *src = bytes.data();
  uint64_t *dst = temp.data();

  for (int byte = 0; byte < 8; ++byte) {
    std::vector<std::vector<int>> local_histograms(num_threads, std::vector<int>(256, 0));
    std::vector<std::thread> threads;
    size_t chunk_size = size / num_threads;

    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t]() {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? size : (t + 1) * chunk_size;
        unsigned char *byte_view = reinterpret_cast<unsigned char *>(src);
        for (size_t i = start; i < end; ++i) {
          local_histograms[t][byte_view[i * 8 + byte]]++;
        }
      });
    }
    for (auto &t : threads) t.join();
    threads.clear();

    std::vector<int> global_histogram(256, 0);
    for (int b = 0; b < 256; ++b) {
      for (int t = 0; t < num_threads; ++t) {
        global_histogram[b] += local_histograms[t][b];
      }
    }

    std::vector<int> global_offsets(256, 0);
    int current_offset = 0;
    for (int b = 0; b < 256; ++b) {
      int count = global_histogram[b];
      global_histogram[b] = current_offset;
      current_offset += count;
    }

    std::vector<std::vector<int>> thread_bucket_offsets(num_threads, std::vector<int>(256, 0));
    for (int b = 0; b < 256; ++b) {
      int running_offset = global_histogram[b];
      for (int t = 0; t < num_threads; ++t) {
        thread_bucket_offsets[t][b] = running_offset;
        running_offset += local_histograms[t][b];
      }
    }

    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t]() {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? size : (t + 1) * chunk_size;
        unsigned char *byte_view = reinterpret_cast<unsigned char *>(src);
        for (size_t i = start; i < end; ++i) {
          int bucket = byte_view[i * 8 + byte];
          dst[thread_bucket_offsets[t][bucket]++] = src[i];
        }
      });
    }
    for (auto &t : threads) t.join();
    std::swap(src, dst);
  }

  for (size_t i = 0; i < size; ++i) {
    arr[i] = FromBytes(src[i]);
  }
}

}  // namespace papulina_y_radix_sort
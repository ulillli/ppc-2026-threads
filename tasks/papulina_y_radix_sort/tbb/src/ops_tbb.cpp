#include "papulina_y_radix_sort/tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
#include "papulina_y_radix_sort/common/include/common.hpp"

namespace papulina_y_radix_sort {

PapulinaYRadixSortTBB::PapulinaYRadixSortTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool PapulinaYRadixSortTBB::ValidationImpl() {
  return true;
}

bool PapulinaYRadixSortTBB::PreProcessingImpl() {
  return true;
}

bool PapulinaYRadixSortTBB::RunImpl() {
  double *result = GetInput().data();
    size_t size = GetInput().size();
    
    ParallelRadixSort(result, static_cast<int>(size));
    
    GetOutput() = std::vector<double>(size);
    for (size_t i = 0; i < size; i++) {
        //std::cout << result[i] << " ";
        GetOutput()[i] = result[i];
    }
    //std::cout << std::endl;
    
    return true;
}

bool PapulinaYRadixSortTBB::PostProcessingImpl() {
  return true;
}
uint64_t PapulinaYRadixSortTBB::InBytes(double d) {
  uint64_t bits = 0;
  memcpy(&bits, &d, sizeof(double));
  if ((bits & kMask) != 0) {
    bits = ~bits;
  } else {
    bits = bits ^ kMask;
  }
  return bits;
}
double PapulinaYRadixSortTBB::FromBytes(uint64_t bits) {
  double d = NAN;
  if ((bits & kMask) != 0) {
    bits = bits ^ kMask;
  } else {
    bits = ~bits;
  }
  memcpy(&d, &bits, sizeof(double));
  return d;
}
void PapulinaYRadixSortTBB::ParallelSortByByte(uint64_t *bytes, uint64_t *out, int byte, int size) {
    auto *byte_view = reinterpret_cast<unsigned char *>(bytes);
    std::array<int, 256> global_counter = {0};
    
    tbb::parallel_for(tbb::blocked_range<int>(0, size),
        [&](const tbb::blocked_range<int>& range) {
            std::array<int, 256> local_counter = {0};
            
            for (int i = range.begin(); i != range.end(); ++i) {
                int index = byte_view[(8 * i) + byte];
                local_counter[index]++;
            }
            
            for (int j = 0; j < 256; j++) {
                if (local_counter[j] != 0) {
                    std::atomic_ref<int> atomic_counter(global_counter[j]);
                    atomic_counter += local_counter[j];
                }
            }
        });
    
    int tmp = 0;
    int j = 0;
    for (; j < 256; j++) {
        if (global_counter[j] != 0) {
            tmp = global_counter[j];
            global_counter[j] = 0;
            j++;
            break;
        }
    }
    for (; j < 256; j++) {
        int a = global_counter[j];
        global_counter[j] = tmp;
        tmp += a;
    }
    
    std::array<std::atomic<int>, 256> atomic_offsets;
    for (int i = 0; i < 256; i++) {
        atomic_offsets[i] = global_counter[i];
    }
    
    tbb::parallel_for(tbb::blocked_range<int>(0, size),
        [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                int index = byte_view[(8 * i) + byte];
                int pos = atomic_offsets[index].fetch_add(1);
                out[pos] = bytes[i];
            }
        });
}
 void PapulinaYRadixSortTBB::ParallelRadixSort(double *arr, int size) {
    std::vector<uint64_t> bytes(size);
    std::vector<uint64_t> out(size);

    tbb::parallel_for(tbb::blocked_range<int>(0, size),
        [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                bytes[i] = InBytes(arr[i]);
            }
        });

    ParallelSortByByte(bytes.data(), out.data(), 0, size);
    ParallelSortByByte(out.data(), bytes.data(), 1, size);
    ParallelSortByByte(bytes.data(), out.data(), 2, size);
    ParallelSortByByte(out.data(), bytes.data(), 3, size);
    ParallelSortByByte(bytes.data(), out.data(), 4, size);
    ParallelSortByByte(out.data(), bytes.data(), 5, size);
    ParallelSortByByte(bytes.data(), out.data(), 6, size);
    ParallelSortByByte(out.data(), bytes.data(), 7, size);
    
    tbb::parallel_for(tbb::blocked_range<int>(0, size),
        [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i != range.end(); ++i) {
                arr[i] = FromBytes(bytes[i]);
            }
        });
}
}  // namespace papulina_y_radix_sort

#pragma once

#include "papulina_y_radix_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace papulina_y_radix_sort {

class PapulinaYRadixSortSTL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSTL;
  }
  explicit PapulinaYRadixSortSTL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  uint64_t InBytes(double d);
  double FromBytes(uint64_t bits);
  void RadixSortParallel(double *arr, size_t size);

};

}  // namespace papulina_y_radix_sort

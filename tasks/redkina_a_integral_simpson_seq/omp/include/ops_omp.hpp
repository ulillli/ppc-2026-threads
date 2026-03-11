// redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp
#pragma once

#include <functional>
#include <vector>

#include "redkina_a_integral_simpson_seq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_integral_simpson_seq {

class RedkinaAIntegralSimpsonOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }

  explicit RedkinaAIntegralSimpsonOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательная функция для вычисления частичной суммы на одном чанке
  double CalculateChunkSum(size_t start_idx, size_t count, const std::vector<double> &h,
                           const std::vector<int> &strides) const;

  std::function<double(const std::vector<double> &)> func_;
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<int> n_;
  double result_ = 0.0;
};

}  // namespace redkina_a_integral_simpson_seq

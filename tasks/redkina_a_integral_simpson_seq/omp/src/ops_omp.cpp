// redkina_a_integral_simpson_seq/omp/src/ops_omp.cpp
#include "redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "redkina_a_integral_simpson_seq/common/include/common.hpp"
#include "util/include/util.hpp"

namespace redkina_a_integral_simpson_seq {

RedkinaAIntegralSimpsonOMP::RedkinaAIntegralSimpsonOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RedkinaAIntegralSimpsonOMP::ValidationImpl() {
  const auto &in = GetInput();
  size_t dim = in.a.size();

  if (dim == 0 || in.b.size() != dim || in.n.size() != dim) {
    return false;
  }

  for (size_t i = 0; i < dim; ++i) {
    if (in.a[i] >= in.b[i]) {
      return false;
    }
    if (in.n[i] <= 0 || in.n[i] % 2 != 0) {
      return false;
    }
  }

  return static_cast<bool>(in.func);
}

bool RedkinaAIntegralSimpsonOMP::PreProcessingImpl() {
  const auto &in = GetInput();
  func_ = in.func;
  a_ = in.a;
  b_ = in.b;
  n_ = in.n;
  result_ = 0.0;
  return true;
}

bool RedkinaAIntegralSimpsonOMP::RunImpl() {
  size_t dim = a_.size();

  // Шаг интегрирования по каждому измерению
  std::vector<double> h(dim);
  double h_prod = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (b_[i] - a_[i]) / static_cast<double>(n_[i]);
    h_prod *= h[i];
  }

  // Множители для линеаризации многомерного индекса (система счисления со смешанными основаниями)
  std::vector<int> strides(dim);
  strides[dim - 1] = 1;
  for (int i = static_cast<int>(dim) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * (n_[i + 1] + 1);
  }
  size_t total_nodes = static_cast<size_t>(strides[0]) * static_cast<size_t>(n_[0] + 1);

  double total_sum = 0.0;

  // Устанавливаем число потоков согласно настройкам
  omp_set_num_threads(ppc::util::GetNumThreads());

#pragma omp parallel default(none) shared(total_nodes, h, strides, dim, h_prod) reduction(+ : total_sum)
  {
    size_t tid = static_cast<size_t>(omp_get_thread_num());
    size_t threads = static_cast<size_t>(omp_get_num_threads());

    // Распределение индексов между потоками (равномерно с учётом остатка)
    size_t chunk = total_nodes / threads;
    size_t remainder = total_nodes % threads;
    size_t my_start = tid * chunk + std::min(tid, remainder);
    size_t my_count = chunk + (tid < remainder ? 1 : 0);

    if (my_count > 0) {
      total_sum += CalculateChunkSum(my_start, my_count, h, strides);
    }
  }

  // Знаменатель 3^dim
  double denominator = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    denominator *= 3.0;
  }

  result_ = (h_prod / denominator) * total_sum;
  return true;
}

double RedkinaAIntegralSimpsonOMP::CalculateChunkSum(size_t start_idx, size_t count, const std::vector<double> &h,
                                                     const std::vector<int> &strides) const {
  size_t dim = a_.size();

  // Восстанавливаем многомерные индексы для стартовой точки
  std::vector<int> indices(dim);
  size_t remainder = start_idx;
  for (size_t d = 0; d < dim; ++d) {
    indices[d] = static_cast<int>(remainder / static_cast<size_t>(strides[d]));
    remainder = remainder % static_cast<size_t>(strides[d]);
  }

  double chunk_sum = 0.0;
  std::vector<double> point(dim);

  for (size_t iter = 0; iter < count; ++iter) {
    // Вычисляем координаты точки
    for (size_t d = 0; d < dim; ++d) {
      point[d] = a_[d] + indices[d] * h[d];
    }

    // Вычисляем произведение весов Симпсона для текущей точки
    double w_prod = 1.0;
    for (size_t d = 0; d < dim; ++d) {
      int idx = indices[d];
      int w;
      if (idx == 0 || idx == n_[d]) {
        w = 1;
      } else if (idx % 2 == 1) {
        w = 4;
      } else {
        w = 2;
      }
      w_prod *= static_cast<double>(w);
    }

    chunk_sum += w_prod * func_(point);

    // Переходим к следующей точке (инкремент многомерного индекса)
    int d = static_cast<int>(dim) - 1;
    while (d >= 0) {
      ++indices[d];
      if (indices[d] <= n_[d]) {
        break;
      }
      indices[d] = 0;
      --d;
    }
  }

  return chunk_sum;
}

bool RedkinaAIntegralSimpsonOMP::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace redkina_a_integral_simpson_seq

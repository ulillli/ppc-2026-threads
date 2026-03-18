#include "peterson_r_graham_scan_omp/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>
#include <vector>

namespace peterson_r_graham_scan_omp {

namespace {
constexpr double kTolerance = 1e-12;

double CalculateOrientation(const Point2D &origin, const Point2D &a, const Point2D &b) {
  return ((a.coord_x - origin.coord_x) * (b.coord_y - origin.coord_y)) -
         ((a.coord_y - origin.coord_y) * (b.coord_x - origin.coord_x));
}

double CalculateSquaredDistance(const Point2D &first, const Point2D &second) {
  const double dx = first.coord_x - second.coord_x;
  const double dy = first.coord_y - second.coord_y;
  return (dx * dx) + (dy * dy);
}

class PointComparator {
 public:
  explicit PointComparator(const Point2D *reference) : origin_ptr_(reference) {}

  bool operator()(const Point2D &lhs, const Point2D &rhs) const {
    const double orientation = CalculateOrientation(*origin_ptr_, lhs, rhs);
    if (std::abs(orientation) > kTolerance) {
      return orientation > 0;
    }
    return CalculateSquaredDistance(*origin_ptr_, lhs) < CalculateSquaredDistance(*origin_ptr_, rhs);
  }

 private:
  const Point2D *origin_ptr_;
};
}  // namespace

PetersonGrahamScannerOMP::PetersonGrahamScannerOMP(const InputValue &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

void PetersonGrahamScannerOMP::LoadPoints(const PointSet &points) {
  input_points_ = points;
  external_data_provided_ = true;
}

PointSet PetersonGrahamScannerOMP::GetConvexHull() const {
  return hull_points_;
}

bool PetersonGrahamScannerOMP::ValidationImpl() {
  return GetInput() >= 0;
}

bool PetersonGrahamScannerOMP::PreProcessingImpl() {
  hull_points_.clear();

  if (!external_data_provided_) {
    input_points_.clear();
    const int count = GetInput();
    if (count <= 0) {
      return true;
    }

    input_points_.resize(count);
    const double angle_step = 2.0 * std::numbers::pi / count;

    // Создаем локальную ссылку для OpenMP
    auto &local_points = input_points_;

#pragma omp parallel for
    for (int i = 0; i < count; ++i) {
      const double angle = angle_step * i;
      local_points[i] = Point2D(std::cos(angle), std::sin(angle));
    }
  }

  return true;
}

bool PetersonGrahamScannerOMP::RunImpl() {
  hull_points_.clear();
  const std::size_t total_points = input_points_.size();

  if (total_points == 0) {
    return true;
  }

  if (AreAllPointsIdentical(input_points_)) {
    hull_points_.push_back(input_points_.front());
    return true;
  }

  if (total_points < 3) {
    hull_points_ = input_points_;
    return true;
  }

  // Параллельный поиск самой нижней точки
  const std::size_t lowest_idx = FindLowestPointParallel(input_points_);
  std::swap(input_points_[0], input_points_[lowest_idx]);

  // Параллельная сортировка по полярному углу
  SortByAngleParallel(input_points_);

  // Последовательный алгоритм Грэхема
  std::vector<Point2D> stack;
  stack.reserve(total_points);
  stack.push_back(input_points_[0]);
  stack.push_back(input_points_[1]);

  for (std::size_t i = 2; i < total_points; ++i) {
    while (stack.size() >= 2) {
      const Point2D &second_last = stack[stack.size() - 2];
      const Point2D &last = stack.back();

      if (CalculateOrientation(second_last, last, input_points_[i]) <= kTolerance) {
        stack.pop_back();
      } else {
        break;
      }
    }
    stack.push_back(input_points_[i]);
  }

  hull_points_ = std::move(stack);
  return true;
}

bool PetersonGrahamScannerOMP::PostProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

double PetersonGrahamScannerOMP::ComputeOrientation(const Point2D &origin, const Point2D &a, const Point2D &b) {
  return CalculateOrientation(origin, a, b);
}

double PetersonGrahamScannerOMP::ComputeDistanceSq(const Point2D &p1, const Point2D &p2) {
  return CalculateSquaredDistance(p1, p2);
}

bool PetersonGrahamScannerOMP::AreAllPointsIdentical(const PointSet &points) {
  if (points.empty()) {
    return true;
  }

  const Point2D &reference = points[0];
  const std::size_t num_points = points.size();

  bool all_identical = true;

  // Создаем локальные ссылки для OpenMP
  const auto &local_points = points;
  const auto &local_ref = reference;

#pragma omp parallel for reduction(&& : all_identical)
  for (std::size_t i = 1; i < num_points; ++i) {
    if (std::abs(local_points[i].coord_x - local_ref.coord_x) > kTolerance ||
        std::abs(local_points[i].coord_y - local_ref.coord_y) > kTolerance) {
      all_identical = false;
    }
  }

  return all_identical;
}

std::size_t PetersonGrahamScannerOMP::FindLowestPointParallel(const PointSet &points) {
  const std::size_t num_points = points.size();
  std::size_t lowest = 0;

  // Создаем локальную ссылку для OpenMP
  const auto &local_points = points;

#pragma omp parallel
  {
    std::size_t local_lowest = 0;

#pragma omp for nowait
    for (std::size_t i = 1; i < num_points; ++i) {
      if (local_points[i].coord_y < local_points[local_lowest].coord_y ||
          (std::abs(local_points[i].coord_y - local_points[local_lowest].coord_y) < kTolerance &&
           local_points[i].coord_x < local_points[local_lowest].coord_x)) {
        local_lowest = i;
      }
    }

#pragma omp critical
    {
      if (local_points[local_lowest].coord_y < local_points[lowest].coord_y ||
          (std::abs(local_points[local_lowest].coord_y - local_points[lowest].coord_y) < kTolerance &&
           local_points[local_lowest].coord_x < local_points[lowest].coord_x)) {
        lowest = local_lowest;
      }
    }
  }

  return lowest;
}

void PetersonGrahamScannerOMP::SortByAngleParallel(PointSet &points) {
  if (points.size() < 2) {
    return;
  }

  const Point2D origin = points[0];

  // Используем параллельную быструю сортировку
  ParallelQuickSort(points, 1, static_cast<int>(points.size()) - 1, origin);
}

int PetersonGrahamScannerOMP::Partition(PointSet &points, int left, int right, const Point2D &origin) {
  Point2D pivot = points[static_cast<std::size_t>(right)];
  int i = left - 1;

  PointComparator comp(&origin);

  for (int j = left; j < right; ++j) {
    if (comp(points[static_cast<std::size_t>(j)], pivot)) {
      i++;
      std::swap(points[static_cast<std::size_t>(i)], points[static_cast<std::size_t>(j)]);
    }
  }

  std::swap(points[static_cast<std::size_t>(i + 1)], points[static_cast<std::size_t>(right)]);
  return i + 1;
}

void PetersonGrahamScannerOMP::ParallelQuickSort(PointSet &points, int left, int right, const Point2D &origin) {
  if (left < right) {
    int pivot = Partition(points, left, right, origin);

    // Создаем локальные ссылки для OpenMP
    auto &local_points = points;
    const auto &local_origin = origin;

#pragma omp parallel sections
    {
#pragma omp section
      {
        ParallelQuickSort(local_points, left, pivot - 1, local_origin);
      }
#pragma omp section
      {
        ParallelQuickSort(local_points, pivot + 1, right, local_origin);
      }
    }
  }
}

}  // namespace peterson_r_graham_scan_omp

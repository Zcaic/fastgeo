#pragma once

#include "fastgeo/core/types.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fastgeo::parallel {

// 设置全局使用的线程数 (可暴露给 Python 端调用，如 fastgeo.set_num_threads(8))
inline void set_num_threads(int num_threads) noexcept {
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void)num_threads;
#endif
}

// 获取当前最大可用线程数
inline int get_max_threads() noexcept {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// 静态调度的并行循环：适用于每个点计算量恒定的场景 (如 B样条/FFD 批量正向求值、导数计算)
template <typename Func>
inline void parallel_for_static(Index begin, Index end, Func&& f, Index min_parallel_size = 256) {
    const Index total = end - begin;
    if (total <= 0) return;

#ifdef _OPENMP
    if (total >= min_parallel_size && omp_get_max_threads() > 1 && !omp_in_parallel()) {
        #pragma omp parallel for schedule(static)
        for (Index i = begin; i < end; ++i) {
            f(i);
        }
        return;
    }
#else
    (void)min_parallel_size;
#endif

    // 数据量较小或未开启 OpenMP 时直接串行执行，避免线程调度开销
    for (Index i = begin; i < end; ++i) {
        f(i);
    }
}

// 动态调度的并行循环：适用于每个点计算耗时不均匀的场景 (如 ADT 搜索 + Newton 迭代求逆投影)
template <typename Func>
inline void parallel_for_dynamic(Index begin, Index end, Func&& f, Index chunk_size = 64) {
    const Index total = end - begin;
    if (total <= 0) return;

#ifdef _OPENMP
    if (total >= chunk_size && omp_get_max_threads() > 1 && !omp_in_parallel()) {
        #pragma omp parallel for schedule(dynamic, 64)
        for (Index i = begin; i < end; ++i) {
            f(i);
        }
        return;
    }
#else
    (void)chunk_size;
#endif

    for (Index i = begin; i < end; ++i) {
        f(i);
    }
}

} // namespace fastgeo::parallel
//
// Copyright 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

/**
 * @brief Contains declarations and definitions for sequential and multi-threading implementations.
 * Multi-threading support is implemented in two variants: using the Threading Building Blocks library and OpenMP* product.
 * To build a particular implementation, use the corresponding identifier: IE_THREAD_TBB, IE_THREAD_OMP or IE_THREAD_SEQ.
 * @file ie_parallel.hpp
 */

#pragma once

#define IE_THREAD_TBB 0
#define IE_THREAD_OMP 1
#define IE_THREAD_SEQ 2

#if IE_THREAD == IE_THREAD_TBB
#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"

inline int  parallel_get_max_threads() { return tbb::this_task_arena::max_concurrency(); }
inline int  parallel_get_num_threads() { return parallel_get_max_threads(); }
inline int  parallel_get_thread_num()  { return tbb::this_task_arena::current_thread_index(); }
inline void parallel_set_num_threads(int n) { return; }

#elif IE_THREAD == IE_THREAD_OMP
#include <omp.h>
/* MSVC still supports omp 2.0 only */
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#   define collapse(x)
#endif  // defined(_MSC_VER) && !defined(__INTEL_COMPILER)
inline int  parallel_get_max_threads() { return omp_get_max_threads(); }
inline int  parallel_get_num_threads() { return omp_get_num_threads(); }
inline int  parallel_get_thread_num()  { return omp_get_thread_num(); }
inline void parallel_set_num_threads(int n) { omp_set_num_threads(n); }

#elif IE_THREAD == IE_THREAD_SEQ
inline int  parallel_get_max_threads() { return 1; }
inline int  parallel_get_num_threads() { return 1; }
inline int  parallel_get_thread_num()  { return 0; }
inline void parallel_set_num_threads(int n) { return; }
#endif


namespace InferenceEngine {

template <typename F>
void parallel_nt(int nthr, F func) {
#if IE_THREAD == IE_THREAD_TBB
    if (nthr == 0) nthr = parallel_get_max_threads();
    if (nthr == 1) {
        func(0, 1);
        return;
    }

    tbb::parallel_for(0, nthr, [&](int ithr) {
        func(ithr, nthr);
    });
#elif IE_THREAD == IE_THREAD_OMP
    if (nthr == 1) {
        func(0, 1);
        return;
    }

#   pragma omp parallel num_threads(nthr)
    func(parallel_get_thread_num(), parallel_get_num_threads());
#elif IE_THREAD == IE_THREAD_SEQ
    func(0, 1);
#endif
}

template <typename T0, typename R, typename F>
R parallel_sum(const T0 D0, R &input, F func) {
#if IE_THREAD == IE_THREAD_TBB
    return tbb::parallel_reduce(
        tbb::blocked_range<T0>(0, D0), input,
        [&](const tbb::blocked_range<T0>& r, R init)->R {
            R sum = init;
            for (T0 dim1 = r.begin(); dim1 < r.end(); ++dim1)
                sum += func(dim1);
            return sum;
        },
        [](R x, R y)->R {
            return x + y;
        });
#else
    R sum = input;
#if IE_THREAD == IE_THREAD_OMP
    #pragma omp parallel for reduction(+ : sum) schedule(static)
#endif
    for (T0 dim1 = 0; dim1 < D0; dim1++) {
        sum += func(dim1);
    }
    return sum;
#endif
}

template <typename T0, typename T1, typename R, typename F>
R parallel_sum2d(const T0 D0, const T1 D1, R input, F func) {
#if IE_THREAD == IE_THREAD_TBB
    return tbb::parallel_reduce(
        tbb::blocked_range2d<T0, T1>(0, D0, 0, D1), input,
        [&](const tbb::blocked_range2d<T0, T1>& r, R init)->R {
            R sum = init;
            for (T0 dim2 = r.rows().begin(); dim2 < r.rows().end(); dim2++) {
                for (T1 dim1 = r.cols().begin(); dim1 < r.cols().end(); dim1++) {
                    sum += func(dim2, dim1);
                }
            }
            return sum;
        },
        [](R x, R y)->R {
            return x + y;
        });
#else
    R sum = input;
#if IE_THREAD == IE_THREAD_OMP
    #pragma omp parallel for collapse(2) reduction(+ : sum) schedule(static)
#endif
    for (T0 dim2 = 0; dim2 < D0; dim2++) {
        for (T1 dim1 = 0; dim1 < D1; dim1++) {
            sum += func(dim2, dim1);
        }
    }
    return sum;
#endif
}

template<typename T>
inline T parallel_it_init(T start) { return start; }
template<typename T, typename Q, typename R, typename... Args>
inline T parallel_it_init(T start, Q &x, const R &X, Args &&... tuple) {
    start = parallel_it_init(start, static_cast<Args>(tuple)...);
    x = start % X;
    return start / X;
}

inline bool parallel_it_step() { return true; }
template<typename Q, typename R, typename... Args>
inline bool parallel_it_step(Q &x, const R &X, Args &&... tuple) {
    if (parallel_it_step(static_cast<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }
    return false;
}

template <typename T, typename Q>
inline void splitter(T n, Q team, Q tid, T &n_start, T &n_end) {
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end = n;
    } else {
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_end = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}


template <typename T0, typename F>
void for_1d(const int ithr, const int nthr, const T0 &D0, F func) {
    T0 d0{ 0 }, end{ 0 };
    splitter(D0, nthr, ithr, d0, end);
    for (; d0 < end; ++d0) func(d0);
}

template <typename T0, typename F>
void parallel_for(const T0 &D0, F func) {
#if IE_THREAD == IE_THREAD_TBB
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_1d(ithr, nthr, D0, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
    #   pragma omp parallel
    for_1d(parallel_get_thread_num(), parallel_get_num_threads(), D0, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_1d(0, 1, D0, func);
#endif
}


template <typename T0, typename T1, typename F>
void for_2d(const int ithr, const int nthr, const T0 &D0, const T1 &D1, F func) {
    const size_t work_amount = (size_t)D0 * D1;
    if (work_amount == 0) return;
    size_t start{ 0 }, end{ 0 };
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{ 0 }; T1 d1{ 0 };
    parallel_it_init(start, d0, D0, d1, D1);
    for (size_t iwork = start; iwork < end; ++iwork) {
        func(d0, d1);
        parallel_it_step(d0, D0, d1, D1);
    }
}

template <typename T0, typename T1, typename F>
void parallel_for2d(const T0 &D0, const T1 &D1, F func) {
#if IE_THREAD == IE_THREAD_TBB
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_2d(ithr, nthr, D0, D1, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
    #   pragma omp parallel
    for_2d(parallel_get_thread_num(), parallel_get_num_threads(), D0, D1, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_2d(0, 1, D0, D1, func);
#endif
}


template <typename T0, typename T1, typename T2, typename F>
void for_3d(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
    const T2 &D2, F func) {
    const size_t work_amount = (size_t)D0 * D1 * D2;
    if (work_amount == 0) return;
    size_t start{ 0 }, end{ 0 };
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{ 0 }; T1 d1{ 0 }; T2 d2{ 0 };
    parallel_it_init(start, d0, D0, d1, D1, d2, D2);
    for (size_t iwork = start; iwork < end; ++iwork) {
        func(d0, d1, d2);
        parallel_it_step(d0, D0, d1, D1, d2, D2);
    }
}

template <typename T0, typename T1, typename T2, typename F>
void parallel_for3d(const T0 &D0, const T1 &D1, const T2 &D2, F func) {
#if IE_THREAD == IE_THREAD_TBB
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_3d(ithr, nthr, D0, D1, D2, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
    #   pragma omp parallel
    for_3d(parallel_get_thread_num(), parallel_get_num_threads(), D0, D1, D2, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_3d(0, 1, D0, D1, D2, func);
#endif
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void for_4d(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
    const T2 &D2, const T3 &D3, F func) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3;
    if (work_amount == 0) return;
    size_t start{ 0 }, end{ 0 };
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{ 0 }; T1 d1{ 0 }; T2 d2{ 0 }; T3 d3{ 0 };
    parallel_it_init(start, d0, D0, d1, D1, d2, D2, d3, D3);
    for (size_t iwork = start; iwork < end; ++iwork) {
        func(d0, d1, d2, d3);
        parallel_it_step(d0, D0, d1, D1, d2, D2, d3, D3);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename F>
void parallel_for4d(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3, F func) {
#if IE_THREAD == IE_THREAD_TBB
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_4d(ithr, nthr, D0, D1, D2, D3, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
    #   pragma omp parallel
    for_4d(parallel_get_thread_num(), parallel_get_num_threads(), D0, D1, D2, D3, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_4d(0, 1, D0, D1, D2, D3, func);
#endif
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename F>
void for_5d(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, F func) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4;
    if (work_amount == 0) return;
    size_t start{ 0 }, end{ 0 };
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{ 0 }; T1 d1{ 0 }; T2 d2{ 0 }; T3 d3{ 0 }; T4 d4{ 0 };
    parallel_it_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    for (size_t iwork = start; iwork < end; ++iwork) {
        func(d0, d1, d2, d3, d4);
        parallel_it_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename F>
void parallel_for5d(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3,
                    const T4 &D4, F func) {
#if IE_THREAD == IE_THREAD_TBB
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_5d(ithr, nthr, D0, D1, D2, D3, D4, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
    #   pragma omp parallel
    for_5d(parallel_get_thread_num(), parallel_get_num_threads(), D0, D1, D2, D3, D4, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_5d(0, 1, D0, D1, D2, D3, D4, func);
#endif
}


template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename F>
void for_6d(const int ithr, const int nthr, const T0 &D0, const T1 &D1,
        const T2 &D2, const T3 &D3, const T4 &D4, const T5 &D5, F func) {
    const size_t work_amount = (size_t)D0 * D1 * D2 * D3 * D4 * D5;
    if (work_amount == 0) return;
    size_t start{ 0 }, end{ 0 };
    splitter(work_amount, nthr, ithr, start, end);

    T0 d0{ 0 }; T1 d1{ 0 }; T2 d2{ 0 }; T3 d3{ 0 }; T4 d4{ 0 }; T5 d5{ 0 };
    parallel_it_init(start, d0, D0, d1, D1, d2, D2, d3, D3, d4, D4,
        d5, D5);
    for (size_t iwork = start; iwork < end; ++iwork) {
        func(d0, d1, d2, d3, d4, d5);
        parallel_it_step(d0, D0, d1, D1, d2, D2, d3, D3, d4, D4, d5, D5);
    }
}

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename F>
void parallel_for6d(const T0 &D0, const T1 &D1, const T2 &D2, const T3 &D3,
    const T4 &D4, const T5 &D5, F func) {
#if IE_THREAD == IE_THREAD_TBB
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        for_6d(ithr, nthr, D0, D1, D2, D3, D4, D5, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
#   pragma omp parallel
    for_6d(parallel_get_thread_num(), parallel_get_num_threads(), D0, D1, D2, D3, D4, D5, func);
#elif IE_THREAD == IE_THREAD_SEQ
    for_6d(0, 1, D0, D1, D2, D3, D4, D5, func);
#endif
}

}  // namespace InferenceEngine


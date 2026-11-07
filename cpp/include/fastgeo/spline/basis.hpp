#pragma once

#include "fastgeo/core/types.hpp"
#include <cassert>
#include <stdexcept>
#include <vector>

namespace fastgeo::spline {

// 支持的最大 B 样条阶数 (order k = degree + 1)，FFD 和 CAD 通常使用 k = 2, 3, 4
inline constexpr int MAX_SPLINE_ORDER = 8;

// ============================================================================
// 1. 寻找参数 u 所在的节点区间下标 (0-based, 对应原 findSpan.f90)
// ============================================================================
/**
 * @brief 二分查找参数 u 在节点向量 knots 中的区间下标 span (0-based)
 * @param u      参数值 (通常在 [knots[k-1], knots[n_ctl]] 之间)
 * @param k      B样条阶数 (order = degree + 1, 如三次样条 k = 4)
 * @param knots  节点向量指针，总长度为 n_ctl + k
 * @param n_ctl  控制点数量
 * @return int   区间下标 span，满足 knots[span] <= u < knots[span+1]，范围在 [k-1, n_ctl-1]
 *               由此可知，非零基函数对应的控制点起始下标为: start_idx = span - k + 1
 */
inline int find_span(Real u, int k, const Real* knots, int n_ctl) noexcept {
    // 处理左边界及越界情况
    if (u <= knots[k - 1]) {
        return k - 1;
    }
    // 处理右边界及越界情况 (与原版 findSpan.f90 的 1-eps 逻辑完全等价且更稳健)
    if (u >= knots[n_ctl]) {
        // 回退跳过末尾重复的节点，确保区间长度 > 0
        int span = n_ctl - 1;
        while (span > k - 1 && knots[span] >= knots[n_ctl]) {
            --span;
        }
        return span;
    }

    // 标准二分查找 (0-based: low = k - 1, high = n_ctl)
    int low = k - 1;
    int high = n_ctl;
    int mid = (low + high) >> 1;

    while (u < knots[mid] || u >= knots[mid + 1]) {
        if (u < knots[mid]) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) >> 1;
    }
    return mid;
}

// Eigen::VectorXd 重载版本
inline int find_span(Real u, int k, const VecX& knots, int n_ctl) noexcept {
    assert(knots.size() == n_ctl + k);
    return find_span(u, k, knots.data(), n_ctl);
}

// ============================================================================
// 2. 计算非零 B 样条基函数值 (0-based, 零堆内存分配, 对应原 basis.f90 -> basis_u)
// ============================================================================
/**
 * @brief 使用 Cox-de Boor 递推公式计算 span 区间上的 k 个非零基函数值
 * @param k      B样条阶数 (1 <= k <= MAX_SPLINE_ORDER)
 * @param span   由 find_span 返回的 0-based 区间下标
 * @param u      参数值
 * @param knots  节点向量指针
 * @param N      [输出] 长度至少为 k 的数组指针，存放 N_{span-k+1, k}(u) ... N_{span, k}(u)
 */
inline void eval_basis(int k, int span, Real u, const Real* knots, Real* N) noexcept {
    assert(k >= 1 && k <= MAX_SPLINE_ORDER);

    Real left[MAX_SPLINE_ORDER]{};
    Real right[MAX_SPLINE_ORDER]{};

    N[0] = 1.0;
    // j 为当前递推的次数 (degree)，从 1 到 k - 1
    for (int j = 1; j < k; ++j) {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        Real saved = 0.0;

        for (int r = 0; r < j; ++r) {
            const Real denom = right[r + 1] + left[j - r];
            const Real temp = (denom != 0.0) ? (N[r] / denom) : 0.0;
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
}

// ============================================================================
// 3. 计算非零 B 样条基函数及其 עד n_deriv 阶导数 (对应原 basis.f90 -> deriv_basis)
// ============================================================================
/**
 * @brief 计算非零基函数及其 1 ~ n_deriv 阶导数 (Piegl & Tiller, The NURBS Book, Alg A2.3)
 * @param k        B样条阶数 (order = degree + 1)
 * @param span     由 find_span 返回的 0-based 区间下标
 * @param u        参数值
 * @param knots    节点向量指针
 * @param n_deriv  需要计算的最高导数阶数 (通常为 1 或 2)
 * @param ders     [输出] 行优先二维数组指针，形状为 (n_deriv + 1) x k
 *                 ders[d * k + j] 表示第 j 个非零基函数的第 d 阶导数 (d=0 为原函数值)
 */
inline void eval_basis_derivs(int k, int span, Real u, const Real* knots, int n_deriv, Real* ders) noexcept {
    assert(k >= 1 && k <= MAX_SPLINE_ORDER);

    // 栈上分配二维表 ndu (k x k) 与工作数组 a (2 x k)
    Real ndu[MAX_SPLINE_ORDER][MAX_SPLINE_ORDER]{};
    Real left[MAX_SPLINE_ORDER]{};
    Real right[MAX_SPLINE_ORDER]{};
    Real a[2][MAX_SPLINE_ORDER]{};

    ndu[0][0] = 1.0;
    for (int j = 1; j < k; ++j) {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        Real saved = 0.0;

        for (int r = 0; r < j; ++r) {
            // 下三角存储节点差 (分母)
            ndu[j][r] = right[r + 1] + left[j - r];
            const Real temp = (ndu[j][r] != 0.0) ? (ndu[r][j - 1] / ndu[j][r]) : 0.0;
            // 上三角存储基函数值
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    // 第 0 行 (d = 0)：直接拷贝 0 阶导数（即基函数值本身）
    for (int j = 0; j < k; ++j) {
        ders[j] = ndu[j][k - 1];
    }

    // 计算 1 ~ n_deriv 阶导数
    const int max_d = (n_deriv < k - 1) ? n_deriv : (k - 1);

    for (int r = 0; r < k; ++r) {
        int s1 = 0, s2 = 1;
        a[0][0] = 1.0;

        for (int d = 1; d <= max_d; ++d) {
            Real d_val = 0.0;
            const int rk = r - d;
            const int pk = (k - 1) - d;

            if (r >= d) {
                a[s2][0] = (ndu[pk + 1][rk] != 0.0) ? (a[s1][0] / ndu[pk + 1][rk]) : 0.0;
                d_val = a[s2][0] * ndu[rk][pk];
            }

            const int j1 = (rk >= -1) ? 1 : -rk;
            const int j2 = (r - 1 <= pk) ? (d - 1) : ((k - 1) - r);

            for (int j = j1; j <= j2; ++j) {
                const Real denom = ndu[pk + 1][rk + j];
                a[s2][j] = (denom != 0.0) ? ((a[s1][j] - a[s1][j - 1]) / denom) : 0.0;
                d_val += a[s2][j] * ndu[rk + j][pk];
            }

            if (r <= pk) {
                const Real denom = ndu[pk + 1][r];
                a[s2][d] = (denom != 0.0) ? (-a[s1][d - 1] / denom) : 0.0;
                d_val += a[s2][d] * ndu[r][pk];
            }

            ders[d * k + r] = d_val;
            std::swap(s1, s2);
        }
    }

    // 乘上阶乘系数 (k-1)! / (k-1-d)!
    int factor = k - 1;
    for (int d = 1; d <= max_d; ++d) {
        for (int j = 0; j < k; ++j) {
            ders[d * k + j] *= static_cast<Real>(factor);
        }
        factor *= (k - 1 - d);
    }

    // 若请求的导数阶数 >= k，高阶导数恒为 0
    for (int d = max_d + 1; d <= n_deriv; ++d) {
        for (int j = 0; j < k; ++j) {
            ders[d * k + j] = 0.0;
        }
    }
}

// ============================================================================
// 4. 节点向量生成算法声明 (对应原 knots.f90，在 basis.cpp 中实现)
// ============================================================================
/**
 * @brief 最小二乘拟合 (LMS) 的节点向量生成算法
 * @param n_data 数据点数量
 * @param n_ctl  控制点数量 (要求 k <= n_ctl <= n_data)
 * @param k      B样条阶数
 * @return VecX  生成的长度为 n_ctl + k 的 0~1 归一化节点向量
 */
VecX knots_lms(int n_data, int n_ctl, int k);

/**
 * @brief 插值 (Interpolation) 的节点向量生成算法
 * @param s      数据点对应的参数值序列 (长度为 n_data)
 * @param s_deriv 导数插值点参数序列 (目前与原版一致，预留参数，默认为空)
 * @param k      B样条阶数
 * @return VecX  生成的长度为 n_ctl + k 的节点向量
 */
VecX knots_interp(const Eigen::Ref<const VecX>& s, const Eigen::Ref<const VecX>& s_deriv, int k);

// ============================================================================
// 5. Boehm 节点插入算法声明 (对应原 insertKnot.f90，在 basis.cpp 中实现)
// ============================================================================
struct KnotInsertionResult {
    int actual_inserted; // 实际成功插入的次数
    int new_n_ctl;       // 插入后的新控制点数量
    VecX new_knots;      // 插入后的新节点向量 (长度: new_n_ctl + k)
    MatX new_coef;       // 插入后的新控制点矩阵 (形状: new_n_ctl x ndim, Row-Major)
};

/**
 * @brief 在一维 B 样条中将节点 u 插入 r 次 (支持任意空间维度 ndim 的控制点)
 * @param u      待插入的参数值 (0 < u < 1)
 * @param r      请求插入的重复次数
 * @param k      B样条阶数 (order)
 * @param knots  原始节点向量 (长度为 n_ctl + k)
 * @param coef   原始控制点矩阵 (形状为 n_ctl x ndim，Row-Major)
 * @return KnotInsertionResult 包含新节点与新控制点的结构体
 */
KnotInsertionResult insert_knot_1d(
    Real u, int r, int k,
    const Eigen::Ref<const VecX>& knots,
    const Eigen::Ref<const MatX>& coef
);

} // namespace fastgeo::spline
// Placeholder
#include "fastgeo/spline/basis.hpp"
#include "fastgeo/core/parallel.hpp"
#include "fastgeo/core/tensor_view.hpp"
#include <algorithm>
#include <cmath>

namespace fastgeo::spline {

// ============================================================================
// 最小二乘节点生成 (严格对标 knots.f90 -> knots_lms)
// ============================================================================
VecX knots_lms(int n_data, int n_ctl, int k) {
    if (n_ctl < k) {
        throw std::invalid_argument("knots_lms: n_ctl must be >= order k.");
    }
    if (n_data < n_ctl) {
        throw std::invalid_argument("knots_lms: n_data must be >= n_ctl.");
    }

    VecX t = VecX::Zero(n_ctl + k);

    // 首尾 k 个重复节点固定为 0.0 和 1.0 (Clamped B-spline)
    t.head(k).setZero();
    t.tail(k).setOnes();

    const int n_inner = n_ctl - k;
    if (n_inner <= 0) {
        return t;
    }

    // 当 k == 2 或 k == 3 时，采用均匀内部节点分布
    if (k == 2 || k == 3) {
        const Real denom = static_cast<Real>(n_ctl - k + 1);
        for (int i = 0; i < n_inner; ++i) {
            t[k + i] = static_cast<Real>(i + 1) / denom;
        }
    } else {
        // 当 k >= 4 时，避免龙格现象 (Runge phenomenon)，两端加密、中间均匀
        if (n_ctl == k) {
            // 无内部节点，已在上方处理
        } else if (n_ctl == k + 1) {
            t[k] = 0.5;
        } else if (n_ctl == k + 2) {
            t[k] = 0.25;
            t[k + 1] = 0.75;
        } else if (n_ctl == k + 3) {
            t[k] = 0.15;
            t[k + 1] = 0.50;
            t[k + 2] = 0.85;
        } else {
            const Real d = 1.0 / static_cast<Real>(n_ctl - k - 2);
            t[k] = 0.15 * d;
            t[k + 1] = 0.50 * d;

            const int n_mid = n_ctl - k - 4;
            for (int i = 0; i < n_mid; ++i) {
                t[k + 2 + i] = static_cast<Real>(i + 1) * d;
            }

            t[n_ctl - 2] = 1.0 - 0.50 * d;
            t[n_ctl - 1] = 1.0 - 0.15 * d;
        }
    }

    return t;
}

// ============================================================================
// 插值节点生成 (严格对标 knots.f90 -> knots_interp，将冒泡排序升级为 std::sort)
// ============================================================================
VecX knots_interp(const Eigen::Ref<const VecX>& s_in, const Eigen::Ref<const VecX>& s_deriv, int k) {
    const int n = static_cast<int>(s_in.size());
    const int nd = static_cast<int>(s_deriv.size());
    if (n < k) {
        throw std::invalid_argument("knots_interp: number of data points must be >= order k.");
    }

    // 拷贝一份参数数组；若未严格递增则执行快速排序 (替代原 Fortran 的 bubble_sort)
    VecX s = s_in;
    if (!std::is_sorted(s.data(), s.data() + n)) {
        std::sort(s.data(), s.data() + n);
    }

    const int n_ctl = n + nd;
    VecX t = VecX::Zero(n_ctl + k);

    // Clamped 两端节点
    t.head(k).setConstant(s[0]);
    t.tail(k).setConstant(s[n - 1]);

    if (nd == 0) {
        // 标准节点平均法 (Knot Averaging, Piegl & Tiller Eq. 9.8)
        const Real inv_deg = 1.0 / static_cast<Real>(k - 1);
        for (int j = 1; j <= n - k; ++j) {
            Real sum = 0.0;
            for (int i = j; i < j + k - 1; ++i) {
                sum += s[i];
            }
            t[k - 1 + j] = sum * inv_deg;
        }
    }

    return t;
}

// ============================================================================
// Boehm 节点插入算法 (严格对标 insertKnot.f90 -> insertknot)
// ============================================================================
KnotInsertionResult insert_knot_1d(
    Real u, int r, int k,
    const Eigen::Ref<const VecX>& knots,
    const Eigen::Ref<const MatX>& coef
) {
    const int n_ctl = static_cast<int>(coef.rows());
    const int ndim = static_cast<int>(coef.cols());
    if (knots.size() != n_ctl + k) {
        throw std::invalid_argument("insert_knot_1d: knots.size() must equal coef.rows() + k.");
    }

    // 1. 找到 u 所在的区间下标 span (0-based, knots[span] <= u < knots[span+1])
    const int span = find_span(u, k, knots.data(), n_ctl);

    // 2. 检查 u 是否已经存在于节点向量中，统计其现有重复度 s
    int s = 0;
    const Real eps = 1e-12;
    for (int i = span; i >= k - 1; --i) {
        if (std::abs(u - knots[i]) <= eps) {
            ++s;
        } else {
            break;
        }
    }

    // 限制插入次数 r，保证插入后的总重复度 s + r <= k - 1 (degree)
    if (r + s > k - 1) {
        r = (k - 1) - s;
    }

    // 若无需插入，直接返回原数据的拷贝
    if (r <= 0) {
        return {0, n_ctl, knots, coef};
    }

    const int new_n_ctl = n_ctl + r;
    VecX t_new(new_n_ctl + k);
    MatX coef_new(new_n_ctl, ndim);

    // 3. 构造新的节点向量 t_new
    for (int i = 0; i <= span; ++i) {
        t_new[i] = knots[i];
    }
    for (int i = 1; i <= r; ++i) {
        t_new[span + i] = u;
    }
    for (int i = span + 1; i < n_ctl + k; ++i) {
        t_new[i + r] = knots[i];
    }

    // 4. 拷贝不受影响的左端和右端控制点
    const int left_end = span - k + 1; // 最后一个不受影响的左端控制点下标
    for (int i = 0; i <= left_end; ++i) {
        coef_new.row(i) = coef.row(i);
    }
    for (int i = span - s; i < n_ctl; ++i) {
        coef_new.row(i + r) = coef.row(i);
    }

    // 5. 使用临时工作矩阵 R (k x ndim) 进行局部递推计算新增/更新的控制点
    MatX R(k, ndim);
    for (int i = 0; i <= k - 1 - s; ++i) {
        R.row(i) = coef.row(span - k + 1 + i);
    }

    int L = 0;
    for (int j = 1; j <= r; ++j) {
        L = span - k + 1 + j;
        for (int i = 0; i <= k - 1 - j - s; ++i) {
            const Real denom = knots[i + span + 1] - knots[L + i];
            const Real alpha = (denom != 0.0) ? ((u - knots[L + i]) / denom) : 0.0;
            R.row(i) = alpha * R.row(i + 1) + (1.0 - alpha) * R.row(i);
        }
        coef_new.row(L) = R.row(0);
        coef_new.row(span + r - j - s) = R.row(k - 1 - j - s);
    }

    // 加载中间剩余的更新控制点
    for (int i = L + 1; i < span - s; ++i) {
        coef_new.row(i) = R.row(i - L);
    }

    return {r, new_n_ctl, std::move(t_new), std::move(coef_new)};
}

} // namespace fastgeo::spline
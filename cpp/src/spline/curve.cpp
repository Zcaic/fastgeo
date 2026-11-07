// Placeholder
#include "fastgeo/spline/curve.hpp"
#include "fastgeo/core/parallel.hpp"
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fastgeo::spline {

// ============================================================================
// 弦长参数化 (严格对标 parameterizations.f90 -> arz)
// ============================================================================
ChordLengthParam chord_length_param(const Eigen::Ref<const MatX>& X) {
    const Index N = X.rows();
    if (N < 2) {
        throw std::invalid_argument("chord_length_param: requires at least 2 points.");
    }

    VecX s(N);
    s[0] = 0.0;
    for (Index i = 1; i < N; ++i) {
        s[i] = s[i - 1] + (X.row(i) - X.row(i - 1)).norm();
    }

    const Real total_len = s[N - 1];
    if (total_len > 0.0) {
        s /= total_len;
    } else {
        // 退化情况：所有输入点重合，回退为均匀参数化
        for (Index i = 0; i < N; ++i) {
            s[i] = static_cast<Real>(i) / static_cast<Real>(N - 1);
        }
    }
    return {total_len, std::move(s)};
}

// ============================================================================
// Curve 构造与拟合 (对标 compute_curve.f90)
// ============================================================================
Curve::Curve(int k, VecX knots, MatX coef)
    : k_(k), knots_(std::move(knots)), coef_(std::move(coef)) {
    if (k_ < 2 || k_ > MAX_SPLINE_ORDER) {
        throw std::invalid_argument("Curve: order k must be in [2, MAX_SPLINE_ORDER].");
    }
    if (knots_.size() != coef_.rows() + k_) {
        throw std::invalid_argument("Curve: knots.size() must equal coef.rows() + k.");
    }
}

std::pair<Curve, VecX> Curve::fit_from_points(
    const Eigen::Ref<const MatX>& X,
    int k,
    int n_ctl,
    const Eigen::Ref<const VecX>& weights,
    const Eigen::Ref<const VecX>& s_in
) {
    const int n_data = static_cast<int>(X.rows());
    const int ndim = static_cast<int>(X.cols());
    if (n_data < k) {
        throw std::invalid_argument("Curve::fit_from_points: n_data must be >= order k.");
    }

    const bool is_interp = (n_ctl <= 0 || n_ctl >= n_data);
    if (is_interp) {
        n_ctl = n_data;
    } else if (n_ctl < k) {
        throw std::invalid_argument("Curve::fit_from_points: n_ctl must be >= order k.");
    }

    // 1. 计算或拷贝参数坐标 s
    VecX s;
    if (s_in.size() == n_data) {
        s = s_in;
    } else {
        s = chord_length_param(X).s;
    }

    // 2. 生成节点向量 knots
    VecX knots = is_interp ? knots_interp(s, VecX(), k) : knots_lms(n_data, n_ctl, k);

    // 3. 特殊极速分支：当 n_data == 2 且 k == 2 时，控制点直接等于两端点
    if (n_data == 2 && k == 2) {
        return {Curve(k, std::move(knots), X), std::move(s)};
    }

    // 4. 组装 B 样条基函数稀疏矩阵 B (形状: n_data x n_ctl)
    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<size_t>(n_data * k));

    Real basis_val[MAX_SPLINE_ORDER];
    for (int i = 0; i < n_data; ++i) {
        const int span = find_span(s[i], k, knots.data(), n_ctl);
        eval_basis(k, span, s[i], knots.data(), basis_val);
        const int start_col = span - k + 1;
        for (int j = 0; j < k; ++j) {
            if (basis_val[j] != 0.0) {
                triplets.emplace_back(i, start_col + j, basis_val[j]);
            }
        }
    }

    Eigen::SparseMatrix<Real, Eigen::ColMajor> B(n_data, n_ctl);
    B.setFromTriplets(triplets.begin(), triplets.end());

    MatX coef(n_ctl, ndim);

// 定义列优先矩阵类型，专用于配合 Eigen::SparseLU 一次性求解多维右端项

    if (is_interp) {
        // 精确插值：直接求解稀疏带状方阵系统 B * coef = X
        Eigen::SparseLU<Eigen::SparseMatrix<Real, Eigen::ColMajor>> solver;
        solver.compute(B);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Curve::fit_from_points: SparseLU factorization failed in interpolation.");
        }
        ColMatX X_col = X;
        ColMatX sol = solver.solve(X_col);
        coef = sol;
    } else {
        // 加权最小二乘拟合：首尾控制点精确锁定两端数据点 (Clamped Boundary Condition)
        coef.row(0) = X.row(0);
        coef.row(n_ctl - 1) = X.row(n_data - 1);

        const int n_inner = n_ctl - 2;
        if (n_inner > 0) {
            VecX W = (weights.size() == n_data) ? weights : VecX::Ones(n_data);

            // 构造从原数据点减去首尾控制点贡献后的右端残差 R (n_data x ndim)
            MatX R = X;
            for (Eigen::SparseMatrix<Real, Eigen::ColMajor>::InnerIterator it(B, 0); it; ++it) {
                R.row(it.row()) -= it.value() * coef.row(0);
            }
            for (Eigen::SparseMatrix<Real, Eigen::ColMajor>::InnerIterator it(B, n_ctl - 1); it; ++it) {
                R.row(it.row()) -= it.value() * coef.row(n_ctl - 1);
            }

            // 截取内部列 B_in (n_data x n_inner)
            Eigen::SparseMatrix<Real, Eigen::ColMajor> B_in = B.middleCols(1, n_inner);
            Eigen::SparseMatrix<Real, Eigen::ColMajor> B_in_T = B_in.transpose();

            // 组装带状对称正定矩阵 A = B_in^T * W * B_in
            Eigen::SparseMatrix<Real, Eigen::ColMajor> A = B_in_T * W.asDiagonal() * B_in;
            ColMatX rhs_col = B_in_T * (W.asDiagonal() * R);

            Eigen::SparseLU<Eigen::SparseMatrix<Real, Eigen::ColMajor>> solver;
            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                throw std::runtime_error("Curve::fit_from_points: SparseLU failed in LMS fitting.");
            }
            ColMatX sol = solver.solve(rhs_col);
            coef.middleRows(1, n_inner) = sol;
        }
    }

    return {Curve(k, std::move(knots), std::move(coef)), std::move(s)};
}

// ============================================================================
// 正向求值与导数求值 (零堆分配内核 + 多线程批处理，对标 eval_curve.f90)
// ============================================================================
void Curve::eval_point(Real s, Real* out_pt) const noexcept {
    const int n_ctl = num_ctl();
    const int dim = ndim();
    const int span = find_span(s, k_, knots_.data(), n_ctl);

    Real N[MAX_SPLINE_ORDER];
    eval_basis(k_, span, s, knots_.data(), N);

    for (int d = 0; d < dim; ++d) {
        out_pt[d] = 0.0;
    }

    const int start = span - k_ + 1;
    const Real* coef_ptr = coef_.data();
    for (int l = 0; l < k_; ++l) {
        const Real w = N[l];
        const Real* pt = &coef_ptr[(start + l) * dim];
        for (int d = 0; d < dim; ++d) {
            out_pt[d] += w * pt[d];
        }
    }
}

VecX Curve::eval_point(Real s) const {
    VecX res(ndim());
    eval_point(s, res.data());
    return res;
}

void Curve::eval_deriv(Real s, Real* out_d1) const noexcept {
    const int n_ctl = num_ctl();
    const int dim = ndim();
    const int span = find_span(s, k_, knots_.data(), n_ctl);

    Real ders[2 * MAX_SPLINE_ORDER];
    eval_basis_derivs(k_, span, s, knots_.data(), 1, ders);
    const Real* dN = &ders[k_]; // 第 1 行对应一阶导数

    for (int d = 0; d < dim; ++d) {
        out_d1[d] = 0.0;
    }

    const int start = span - k_ + 1;
    const Real* coef_ptr = coef_.data();
    for (int l = 0; l < k_; ++l) {
        const Real w = dN[l];
        const Real* pt = &coef_ptr[(start + l) * dim];
        for (int d = 0; d < dim; ++d) {
            out_d1[d] += w * pt[d];
        }
    }
}

VecX Curve::eval_deriv(Real s) const {
    VecX res(ndim());
    eval_deriv(s, res.data());
    return res;
}

void Curve::eval_second_deriv(Real s, Real* out_d2) const noexcept {
    const int n_ctl = num_ctl();
    const int dim = ndim();
    const int span = find_span(s, k_, knots_.data(), n_ctl);

    Real ders[3 * MAX_SPLINE_ORDER];
    eval_basis_derivs(k_, span, s, knots_.data(), 2, ders);
    const Real* d2N = &ders[2 * k_]; // 第 2 行对应二阶导数

    for (int d = 0; d < dim; ++d) {
        out_d2[d] = 0.0;
    }

    const int start = span - k_ + 1;
    const Real* coef_ptr = coef_.data();
    for (int l = 0; l < k_; ++l) {
        const Real w = d2N[l];
        const Real* pt = &coef_ptr[(start + l) * dim];
        for (int d = 0; d < dim; ++d) {
            out_d2[d] += w * pt[d];
        }
    }
}

VecX Curve::eval_second_deriv(Real s) const {
    VecX res(ndim());
    eval_second_deriv(s, res.data());
    return res;
}

void Curve::eval_point_and_derivs(Real s, Real* val, Real* d1, Real* d2) const noexcept {
    const int n_ctl = num_ctl();
    const int dim = ndim();
    const int span = find_span(s, k_, knots_.data(), n_ctl);

    Real ders[3 * MAX_SPLINE_ORDER];
    eval_basis_derivs(k_, span, s, knots_.data(), 2, ders);

    for (int d = 0; d < dim; ++d) {
        val[d] = 0.0;
        d1[d] = 0.0;
        d2[d] = 0.0;
    }

    const int start = span - k_ + 1;
    const Real* coef_ptr = coef_.data();
    for (int l = 0; l < k_; ++l) {
        const Real w0 = ders[l];
        const Real w1 = ders[k_ + l];
        const Real w2 = ders[2 * k_ + l];
        const Real* pt = &coef_ptr[(start + l) * dim];
        for (int d = 0; d < dim; ++d) {
            val[d] += w0 * pt[d];
            d1[d]  += w1 * pt[d];
            d2[d]  += w2 * pt[d];
        }
    }
}

void Curve::eval_points_batch(const Real* s_arr, Index n_pts, Real* out_pts) const {
    const int dim = ndim();
    parallel::parallel_for_static(0, n_pts, [&](Index i) {
        eval_point(s_arr[i], &out_pts[i * dim]);
    });
}

MatX Curve::eval_points_batch(const Eigen::Ref<const VecX>& s_arr) const {
    MatX res(s_arr.size(), ndim());
    eval_points_batch(s_arr.data(), s_arr.size(), res.data());
    return res;
}

void Curve::eval_derivs_batch(const Real* s_arr, Index n_pts, Real* out_derivs) const {
    const int dim = ndim();
    parallel::parallel_for_static(0, n_pts, [&](Index i) {
        eval_deriv(s_arr[i], &out_derivs[i * dim]);
    });
}

MatX Curve::eval_derivs_batch(const Eigen::Ref<const VecX>& s_arr) const {
    MatX res(s_arr.size(), ndim());
    eval_derivs_batch(s_arr.data(), s_arr.size(), res.data());
    return res;
}

void Curve::eval_second_derivs_batch(const Real* s_arr, Index n_pts, Real* out_derivs2) const {
    const int dim = ndim();
    parallel::parallel_for_static(0, n_pts, [&](Index i) {
        eval_second_deriv(s_arr[i], &out_derivs2[i * dim]);
    });
}

MatX Curve::eval_second_derivs_batch(const Eigen::Ref<const VecX>& s_arr) const {
    MatX res(s_arr.size(), ndim());
    eval_second_derivs_batch(s_arr.data(), s_arr.size(), res.data());
    return res;
}

// ============================================================================
// 弧长积分、节点插入与方向反转 (对标 pyCurve.py)
// ============================================================================
Real Curve::calc_length() const {
    // 4点 Gauss-Legendre 积分节点与权重 ([-1, 1] 区间)
    static constexpr Real gauss_pts[4] = {
        -0.8611363115940526, -0.3399810435848563,
         0.3399810435848563,  0.8611363115940526
    };
    static constexpr Real gauss_wts[4] = {
         0.3478548451374538,  0.6521451548625461,
         0.6521451548625461,  0.3478548451374538
    };

    const int n_ctl = num_ctl();
    const int dim = ndim();
    Real total_length = 0.0;
    std::vector<Real> d1(dim);

    // 遍历所有有效节点区间 [knots[i], knots[i+1]]
    for (int i = k_ - 1; i < n_ctl; ++i) {
        const Real a = knots_[i];
        const Real b = knots_[i + 1];
        if (b <= a) continue;

        const Real half_len = 0.5 * (b - a);
        const Real mid = 0.5 * (a + b);

        Real span_len = 0.0;
        for (int g = 0; g < 4; ++g) {
            const Real s = mid + half_len * gauss_pts[g];
            eval_deriv(s, d1.data());
            Real norm_sq = 0.0;
            for (int d = 0; d < dim; ++d) {
                norm_sq += d1[d] * d1[d];
            }
            span_len += gauss_wts[g] * std::sqrt(norm_sq);
        }
        total_length += half_len * span_len;
    }

    return total_length;
}

int Curve::insert_knot(Real u, int r) {
    auto res = insert_knot_1d(u, r, k_, knots_, coef_);
    if (res.actual_inserted > 0) {
        knots_ = std::move(res.new_knots);
        coef_ = std::move(res.new_coef);
        invalidate_cache();
    }
    return res.actual_inserted;
}

void Curve::reverse() {
    // 控制点行逆序
    const int n_ctl = num_ctl();
    for (int i = 0; i < n_ctl / 2; ++i) {
        coef_.row(i).swap(coef_.row(n_ctl - 1 - i));
    }
    // 节点向量反转: t_new[i] = t_max + t_min - t_old[M - 1 - i]
    const Real t_sum = knots_[0] + knots_[knots_.size() - 1];
    knots_ = (t_sum - knots_.reverse().array()).matrix();
    invalidate_cache();
}

// ============================================================================
// 逆向投影计算 (对标 projections.F90 -> project_curve & point_curve_start)
// ============================================================================
void Curve::ensure_projection_cache(int n_samples) const {
    if (cache_valid_ && sample_s_.size() == n_samples) {
        return;
    }
    const Real s_min = knots_[k_ - 1];
    const Real s_max = knots_[num_ctl()];
    sample_s_ = VecX::LinSpaced(n_samples, s_min, s_max);
    sample_pts_ = eval_points_batch(sample_s_);
    cache_valid_ = true;
}

void Curve::project_points(
    const Real* pts_ptr, Index n_pts,
    Real* s_inout, Real* diff_out,
    int max_iter, Real tol,
    bool use_guess
) const {
    if (!use_guess) {
        ensure_projection_cache(200);
    }

    const int dim = ndim();
    const Real s_min = knots_[k_ - 1];
    const Real s_max = knots_[num_ctl()];
    const Index n_samples = sample_s_.size();

    parallel::parallel_for_dynamic(0, n_pts, [&](Index idx) {
        const Real* X = &pts_ptr[idx * dim];
        Real s = s_inout[idx];

        // 1. 若未使用外部初值或初值越界，通过预采样折线寻找全局最近参数初值
        if (!use_guess || s < s_min || s > s_max) {
            Real min_dist_sq = std::numeric_limits<Real>::max();
            Index best_k = 0;
            for (Index k = 0; k < n_samples; ++k) {
                Real dist_sq = 0.0;
                const Real* sp = &sample_pts_(k, 0);
                for (int d = 0; d < dim; ++d) {
                    const Real diff = X[d] - sp[d];
                    dist_sq += diff * diff;
                }
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_k = k;
                }
            }
            s = sample_s_[best_k];
        }

        // 2. Newton-Raphson 迭代求解正交条件: f(s) = (C(s) - X) \cdot C'(s) = 0
        //    f'(s) = C'(s) \cdot C'(s) + (C(s) - X) \cdot C''(s)
        //    为支持 ndim <= 16 的栈分配，使用固定上限数组
        Real val[16], d1[16], d2[16];

        for (int iter = 0; iter < max_iter; ++iter) {
            eval_point_and_derivs(s, val, d1, d2);

            Real f = 0.0;
            Real df = 0.0;
            Real dist_sq = 0.0;
            Real d1_norm_sq = 0.0;

            for (int d = 0; d < dim; ++d) {
                const Real diff = val[d] - X[d]; // C(s) - X
                f += diff * d1[d];
                df += d1[d] * d1[d] + diff * d2[d];
                dist_sq += diff * diff;
                d1_norm_sq += d1[d] * d1[d];
            }

            // 点已恰好落在曲线上，或切向导数为 0
            if (dist_sq < tol * tol || d1_norm_sq == 0.0) {
                break;
            }
            // 余弦正交收敛准则: |(C(s) - X) \cdot C'(s)| / (||C(s)-X|| * ||C'(s)||) < tol
            if (std::abs(f) / (std::sqrt(dist_sq) * std::sqrt(d1_norm_sq)) < tol) {
                break;
            }

            // 若二阶导导致海森矩阵非正定 (df <= 0)，退化为 Gauss-Newton 步 (仅保留 d1_norm_sq)
            const Real denom = (df > 0.0) ? df : d1_norm_sq;
            Real ds = -f / denom;

            Real s_next = std::clamp(s + ds, s_min, s_max);
            const Real actual_ds = s_next - s;
            s = s_next;

            if (std::abs(actual_ds) < tol) {
                break;
            }
        }

        // 3. 记录最终收敛参数 s 以及距离差向量 D = X - C(s) (与原版 pyspline 符号约定完全一致)
        eval_point(s, val);
        s_inout[idx] = s;
        Real* out_D = &diff_out[idx * dim];
        for (int d = 0; d < dim; ++d) {
            out_D[d] = X[d] - val[d];
        }
    });
}

std::pair<VecX, MatX> Curve::project_points(
    const Eigen::Ref<const MatX>& pts,
    int max_iter, Real tol,
    const Eigen::Ref<const VecX>& s_guess
) const {
    if (pts.cols() != ndim()) {
        throw std::invalid_argument("Curve::project_points: pts.cols() must match curve ndim().");
    }
    const Index n_pts = pts.rows();
    const bool use_guess = (s_guess.size() == n_pts);
    VecX s_out = use_guess ? s_guess : VecX::Zero(n_pts);
    MatX diff_out(n_pts, ndim());

    project_points(pts.data(), n_pts, s_out.data(), diff_out.data(), max_iter, tol, use_guess);
    return {std::move(s_out), std::move(diff_out)};
}

std::tuple<Real, Real, VecX> Curve::project_curve(
    const Curve& other,
    int max_iter, Real tol,
    Real s_guess, Real t_guess
) const {
    if (ndim() != other.ndim()) {
        throw std::invalid_argument("Curve::project_curve: both curves must have the same ndim().");
    }
    const int dim = ndim();
    const Real s_min = knots_[k_ - 1], s_max = knots_[num_ctl()];
    const Real t_min = other.knots_[other.k_ - 1], t_max = other.knots_[other.num_ctl()];

    Real s = s_guess;
    Real t = t_guess;

    // 1. 若未提供有效初值，使用双曲线离散采样网格搜索全局最近点对
    if (s < s_min || s > s_max || t < t_min || t > t_max) {
        ensure_projection_cache(100);
        other.ensure_projection_cache(100);

        Real min_dist_sq = std::numeric_limits<Real>::max();
        for (Index i = 0; i < sample_s_.size(); ++i) {
            for (Index j = 0; j < other.sample_s_.size(); ++j) {
                const Real dist_sq = (sample_pts_.row(i) - other.sample_pts_.row(j)).squaredNorm();
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    s = sample_s_[i];
                    t = other.sample_s_[j];
                }
            }
        }
    }

    // 2. 二维 Newton-Raphson 求解联立方程组:
    //    R0(s, t) = (C1(s) - C2(t)) \cdot C1'(s) = 0
    //    R1(s, t) = -(C1(s) - C2(t)) \cdot C2'(t) = 0
    VecX c1(dim), dc1(dim), d2c1(dim);
    VecX c2(dim), dc2(dim), d2c2(dim);

    for (int iter = 0; iter < max_iter; ++iter) {
        eval_point_and_derivs(s, c1.data(), dc1.data(), d2c1.data());
        other.eval_point_and_derivs(t, c2.data(), dc2.data(), d2c2.data());

        const VecX diff = c1 - c2;
        if (diff.squaredNorm() < tol * tol) {
            break; // 两曲线相交
        }

        Vec2 R;
        R[0] =  diff.dot(dc1);
        R[1] = -diff.dot(dc2);

        if (R.norm() < tol) {
            break;
        }

        Mat2 J;
        J(0, 0) = dc1.squaredNorm() + diff.dot(d2c1);
        J(0, 1) = -dc1.dot(dc2);
        J(1, 0) = J(0, 1);
        J(1, 1) = dc2.squaredNorm() - diff.dot(d2c2);

        // 求解 2x2 线性系统 J * delta = -R
        Vec2 delta = J.ldlt().solve(-R);

        const Real s_next = std::clamp(s + delta[0], s_min, s_max);
        const Real t_next = std::clamp(t + delta[1], t_min, t_max);

        const Real step_norm = std::hypot(s_next - s, t_next - t);
        s = s_next;
        t = t_next;

        if (step_norm < tol) {
            break;
        }
    }

    VecX final_diff = eval_point(s) - other.eval_point(t);
    return {s, t, std::move(final_diff)};
}

} // namespace fastgeo::spline
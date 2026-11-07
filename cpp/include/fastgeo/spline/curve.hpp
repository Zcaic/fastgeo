#pragma once

#include "fastgeo/core/types.hpp"
#include "fastgeo/spline/basis.hpp"
#include <tuple>
#include <vector>

namespace fastgeo::spline {

/**
 * @brief 弦长参数化结果结构体 (对应原 parameterizations.f90 -> arz)
 */
struct ChordLengthParam {
    Real total_length; // 折线总弦长
    VecX s;            // 归一化弧长参数 s \in [0, 1]，长度为 N
};

/**
 * @brief 计算有序空间点集 (N x ndim) 的归一化累积弦长参数 s
 */
ChordLengthParam chord_length_param(const Eigen::Ref<const MatX>& X);

/**
 * @brief 一维 B 样条曲线核心类 (支持任意空间维度 ndim，如 1D, 2D, 3D)
 *        内部控制点 coef_ 严格采用 Row-Major (n_ctl x ndim) 存储
 */
class Curve {
public:
    Curve() = default;

    /**
     * @brief 构造函数 1：直接通过已知阶数、节点向量和控制点构造曲线
     * @param k      B样条阶数 (order = degree + 1, 例如三次样条 k = 4)
     * @param knots  节点向量 (长度为 n_ctl + k)
     * @param coef   控制点矩阵 (形状为 n_ctl x ndim, Row-Major)
     */
    Curve(int k, VecX knots, MatX coef);

    /**
     * @brief 静态工厂函数：通过数据点插值或最小二乘拟合构造 B 样条曲线 (对应 compute_curve.f90)
     * @param X       输入数据点矩阵 (n_data x ndim)
     * @param k       B样条阶数 (2 <= k <= MAX_SPLINE_ORDER)
     * @param n_ctl   控制点数量 (若 n_ctl >= n_data 或 n_ctl <= 0，则自动执行精确插值 n_ctl = n_data)
     * @param weights 最小二乘权重向量 (可选，长度 n_data，默认全为 1.0)
     * @param s_in    预设参数坐标 (可选，长度 n_data；若为空则自动按弦长参数化计算)
     * @return std::pair<Curve, VecX> 返回构造好的 Curve 对象及对应的参数向量 s
     */
    static std::pair<Curve, VecX> fit_from_points(
        const Eigen::Ref<const MatX>& X,
        int k = 4,
        int n_ctl = -1,
        const Eigen::Ref<const VecX>& weights = VecX(),
        const Eigen::Ref<const VecX>& s_in = VecX()
    );

    // 基础属性访问
    int order() const noexcept { return k_; }
    int degree() const noexcept { return k_ - 1; }
    int num_ctl() const noexcept { return static_cast<int>(coef_.rows()); }
    int ndim() const noexcept { return static_cast<int>(coef_.cols()); }

    const VecX& knots() const noexcept { return knots_; }
    const MatX& coef() const noexcept { return coef_; }
    MatX& coef() noexcept { invalidate_cache(); return coef_; }

    // ========================================================================
    // 1. 单点与批量正向求值 / 导数求值 (对应 eval_curve.f90，支持零拷贝裸指针与多线程)
    // ========================================================================
    /// 单点求值 (写入长度为 ndim 的 out_pt)
    void eval_point(Real s, Real* out_pt) const noexcept;
    VecX eval_point(Real s) const;

    /// 单点一阶导数 dC/ds
    void eval_deriv(Real s, Real* out_d1) const noexcept;
    VecX eval_deriv(Real s) const;

    /// 单点二阶导数 d^2C/ds^2
    void eval_second_deriv(Real s, Real* out_d2) const noexcept;
    VecX eval_second_deriv(Real s) const;

    /// 单点同时求取 0阶(位置)、1阶导数、2阶导数 (在 Newton 投影迭代中极度高频调用，仅查一次 span)
    void eval_point_and_derivs(Real s, Real* val, Real* d1, Real* d2) const noexcept;

    /// 多线程批量求值：输入 N 个参数 s_arr，输出 (N x ndim) 的行优先坐标数组 out_pts
    void eval_points_batch(const Real* s_arr, Index n_pts, Real* out_pts) const;
    MatX eval_points_batch(const Eigen::Ref<const VecX>& s_arr) const;

    /// 多线程批量一阶导数求值
    void eval_derivs_batch(const Real* s_arr, Index n_pts, Real* out_derivs) const;
    MatX eval_derivs_batch(const Eigen::Ref<const VecX>& s_arr) const;

    /// 多线程批量二阶导数求值
    void eval_second_derivs_batch(const Real* s_arr, Index n_pts, Real* out_derivs2) const;
    MatX eval_second_derivs_batch(const Eigen::Ref<const VecX>& s_arr) const;

    // ========================================================================
    // 2. 几何运算：弧长计算、节点插入、方向反转、区間切分
    // ========================================================================
    /// 使用 4 点 Gauss-Legendre 积分计算每个有效节点区间的弧长及总弧长 (对应 pyCurve.calcLength)
    Real calc_length() const;

    /// 插入节点 u 共 r 次，直接原地更新当前曲线并返回实际插入次数
    int insert_knot(Real u, int r);

    /// 反转曲线参数方向 (s -> 1 - s)
    void reverse();

    // ========================================================================
    // 3. 逆向投影计算 (Point / Curve Projection, 对应 projections.F90)
    // ========================================================================
    /**
     * @brief 多线程批量将空间点集 X (N x ndim) 投影到曲线上，求取最近点参数 s 与距离差向量 D = X - C(s)
     * @param pts_ptr   输入空间点首地址 (N x ndim, Row-Major)
     * @param n_pts     点的数量 N
     * @param s_inout   [输入/输出] 长度为 N 的参数数组；若 use_guess=false，内部自动通过离散采样寻找最佳初值
     * @param diff_out  [输出] 形状为 (N x ndim) 的差向量 D = X - C(s)
     * @param max_iter  Newton-Raphson 最大迭代步数
     * @param tol       参数收敛容差 eps_s
     * @param use_guess 是否直接使用 s_inout 中的值作为牛顿迭代初值
     */
    void project_points(
        const Real* pts_ptr, Index n_pts,
        Real* s_inout, Real* diff_out,
        int max_iter = 25, Real tol = 1e-10,
        bool use_guess = false
    ) const;

    std::pair<VecX, MatX> project_points(
        const Eigen::Ref<const MatX>& pts,
        int max_iter = 25, Real tol = 1e-10,
        const Eigen::Ref<const VecX>& s_guess = VecX()
    ) const;

    /**
     * @brief 计算当前曲线 C1(s) 与另一条曲线 C2(t) 之间的最近距离点对 (对应 projections.F90 -> project_curve)
     * @return std::tuple<Real, Real, VecX> 返回 (最优 s, 最优 t, 差向量 D = C1(s) - C2(t))
     */
    std::tuple<Real, Real, VecX> project_curve(
        const Curve& other,
        int max_iter = 25, Real tol = 1e-10,
        Real s_guess = -1.0, Real t_guess = -1.0
    ) const;

    /// 预计算用于投影初值搜索的离散采样缓存 (线程安全)
    void ensure_projection_cache(int n_samples = 200) const;
    void invalidate_cache() noexcept { cache_valid_ = false; }

private:
    int k_ = 0;
    VecX knots_;
    MatX coef_;

    // 用于快速初值查找的离散采样点缓存
    mutable bool cache_valid_ = false;
    mutable VecX sample_s_;
    mutable MatX sample_pts_;
};

} // namespace fastgeo::spline
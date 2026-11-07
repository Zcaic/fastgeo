// Placeholder
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>

#include "fastgeo/spline/basis.hpp"
#include "fastgeo/spline/curve.hpp"

namespace nb = nanobind;
using namespace fastgeo;
using namespace fastgeo::spline;

// 定义常用的零拷贝 NumPy 数组类型别名 (C-contiguous, CPU)
using NdArray1DConst = nb::ndarray<const Real, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NdArray1DMut   = nb::ndarray<Real, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NdArray2DConst = nb::ndarray<const Real, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using NdArray2DMut   = nb::ndarray<Real, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

void bind_spline(nb::module_& m) {
    // ========================================================================
    // 1. 绑定基础节点与参数化工具函数
    // ========================================================================
    m.def("knots_lms", &knots_lms,
          nb::arg("n_data"), nb::arg("n_ctl"), nb::arg("k"),
          "Generate B-spline knot vector for Least-Squares (LMS) fitting.");

    m.def("knots_interp", [](const Eigen::Ref<const VecX>& s, int k) {
        return knots_interp(s, VecX(), k);
    }, nb::arg("s"), nb::arg("k"), "Generate B-spline knot vector for interpolation.");

    m.def("chord_length_param", [](const Eigen::Ref<const MatX>& X) {
        auto res = chord_length_param(X);
        return std::make_pair(res.total_length, std::move(res.s));
    }, nb::arg("X"), "Compute normalized chord-length parameterization (returns total_length, s).");

    // ========================================================================
    // 2. 绑定 Curve 类
    // ========================================================================
    nb::class_<Curve>(m, "Curve")
        .def(nb::init<int, VecX, MatX>(),
             nb::arg("k"), nb::arg("knots"), nb::arg("coef"),
             "Construct a B-spline Curve from order k, knots (1D), and control points coef (n_ctl x ndim).")

        // 静态工厂方法：从数据点插值或最小二乘拟合
        .def_static("fit_from_points", [](
            const Eigen::Ref<const MatX>& X,
            int k,
            int n_ctl,
            const Eigen::Ref<const VecX>& weights,
            const Eigen::Ref<const VecX>& s_in
        ) {
            nb::gil_scoped_release release;
            return Curve::fit_from_points(X, k, n_ctl, weights, s_in);
        },
        nb::arg("X"),
        nb::arg("k") = 4,
        nb::arg("n_ctl") = -1,
        nb::arg("weights") = VecX(),
        nb::arg("s") = VecX(),
        "Fit or interpolate a B-spline Curve from data points X. Returns (Curve, s).")

        // 只读属性与数据访问
        .def_prop_ro("k", &Curve::order, "B-spline order k (degree + 1).")
        .def_prop_ro("degree", &Curve::degree, "B-spline degree (k - 1).")
        .def_prop_ro("num_ctl", &Curve::num_ctl, "Number of control points.")
        .def_prop_ro("ndim", &Curve::ndim, "Spatial dimension of the curve.")
        .def_prop_ro("knots", &Curve::knots, "Knot vector.")
        .def_prop_rw("coef",
            [](const Curve& self) -> const MatX& { return self.coef(); },
            [](Curve& self, const Eigen::Ref<const MatX>& new_coef) {
                if (new_coef.rows() != self.num_ctl() || new_coef.cols() != self.ndim()) {
                    throw std::invalid_argument("New coef shape must match existing (num_ctl, ndim).");
                }
                self.coef() = new_coef;
            },
            "Control points matrix (num_ctl x ndim, Row-Major).")

        // 单点求值与导数
        .def("eval_point", nb::overload_cast<Real>(&Curve::eval_point, nb::const_), nb::arg("s"))
        .def("eval_deriv", nb::overload_cast<Real>(&Curve::eval_deriv, nb::const_), nb::arg("s"))
        .def("eval_second_deriv", nb::overload_cast<Real>(&Curve::eval_second_deriv, nb::const_), nb::arg("s"))

        // 多线程批量求值 (自动分配返回 NumPy 数组，释放 GIL)
        .def("eval_points", [](const Curve& self, const Eigen::Ref<const VecX>& s_arr) {
            nb::gil_scoped_release release;
            return self.eval_points_batch(s_arr);
        }, nb::arg("s"))

        .def("eval_derivs", [](const Curve& self, const Eigen::Ref<const VecX>& s_arr) {
            nb::gil_scoped_release release;
            return self.eval_derivs_batch(s_arr);
        }, nb::arg("s"))

        .def("eval_second_derivs", [](const Curve& self, const Eigen::Ref<const VecX>& s_arr) {
            nb::gil_scoped_release release;
            return self.eval_second_derivs_batch(s_arr);
        }, nb::arg("s"))

        // 零拷贝原地批量求值 (直接写入预分配的 C-contiguous NumPy 数组)
        .def("eval_points_inplace", [](const Curve& self, NdArray1DConst s_arr, NdArray2DMut out_pts) {
            const Index n_pts = static_cast<Index>(s_arr.shape(0));
            if (static_cast<Index>(out_pts.shape(0)) != n_pts || static_cast<int>(out_pts.shape(1)) != self.ndim()) {
                throw std::invalid_argument("out_pts shape must be (len(s), ndim).");
            }
            const Real* s_ptr = s_arr.data();
            Real* out_ptr = out_pts.data();

            nb::gil_scoped_release release;
            self.eval_points_batch(s_ptr, n_pts, out_ptr);
        }, nb::arg("s"), nb::arg("out_pts"), "Zero-copy multithreaded evaluation into pre-allocated out_pts.")

        // 几何工具：弧长、节点插入、方向反转
        .def("calc_length", [](const Curve& self) {
            nb::gil_scoped_release release;
            return self.calc_length();
        })
        .def("insert_knot", &Curve::insert_knot, nb::arg("u"), nb::arg("r") = 1)
        .def("reverse", &Curve::reverse)

        // 逆向投影：点集投影 (多线程) 與曲线间最近距离投影
        .def("project_points", [](
            const Curve& self,
            const Eigen::Ref<const MatX>& pts,
            int max_iter,
            Real tol,
            const Eigen::Ref<const VecX>& s_guess
        ) {
            // 先在主线程确保投影采样缓存已建立，再释放 GIL 进入多线程牛顿迭代
            self.ensure_projection_cache(200);
            nb::gil_scoped_release release;
            return self.project_points(pts, max_iter, tol, s_guess);
        },
        nb::arg("pts"),
        nb::arg("max_iter") = 25,
        nb::arg("tol") = 1e-10,
        nb::arg("s_guess") = VecX(),
        "Multithreaded point projection onto curve. Returns (s, diff_vectors).")

        .def("project_curve", [](
            const Curve& self,
            const Curve& other,
            int max_iter,
            Real tol,
            Real s_guess,
            Real t_guess
        ) {
            self.ensure_projection_cache(100);
            other.ensure_projection_cache(100);
            nb::gil_scoped_release release;
            return self.project_curve(other, max_iter, tol, s_guess, t_guess);
        },
        nb::arg("other"),
        nb::arg("max_iter") = 25,
        nb::arg("tol") = 1e-10,
        nb::arg("s_guess") = -1.0,
        nb::arg("t_guess") = -1.0,
        "Find minimum distance between two curves. Returns (s, t, diff_vector).");
}
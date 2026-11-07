#include "fastgeo/core/parallel.hpp"
#include "fastgeo/spline/curve.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace fastgeo;
using namespace fastgeo::spline;

#define EXPECT_NEAR(val1, val2, tol, msg)                                                   \
    do {                                                                                    \
        if (std::abs((val1) - (val2)) > (tol)) {                                            \
            std::cerr << "[FAILED] " << (msg) << " | Expected: " << (val2)                  \
                      << ", Actual: " << (val1) << ", Diff: " << std::abs((val1) - (val2))  \
                      << std::endl;                                                         \
            return 1;                                                                       \
        }                                                                                   \
    } while (0)

int main() {
    std::cout << "========================================================\n";
    std::cout << "Running FastGeo Spline Core & Curve Module Tests...\n";
    std::cout << "OpenMP Max Threads: " << parallel::get_max_threads() << "\n";
    std::cout << "========================================================\n";

    // ------------------------------------------------------------------------
    // 测试 1：精确插值与正向求值/求导 (拟合三次多项式曲线 x=t, y=t^2, z=t^3)
    // ------------------------------------------------------------------------
    const int n_data = 20;
    MatX pts(n_data, 3);
    VecX t_param = VecX::LinSpaced(n_data, 0.0, 1.0);
    for (int i = 0; i < n_data; ++i) {
        const Real t = t_param[i];
        pts(i, 0) = t;
        pts(i, 1) = t * t;
        pts(i, 2) = t * t * t;
    }

    // 使用 k=4 (三次 B 样条) 进行精确插值，因为三次样条能以机器精度精确表示三次多项式！
    auto [crv, s_out] = Curve::fit_from_points(pts, 4, -1, VecX(), t_param);

    // 验证 s = 0.35 处的位置、一阶导数、二阶导数解析解
    const Real s_test = 0.35;
    VecX val = crv.eval_point(s_test);
    VecX d1  = crv.eval_deriv(s_test);
    VecX d2  = crv.eval_second_deriv(s_test);

    EXPECT_NEAR(val[0], s_test, 1e-13, "Curve eval X");
    EXPECT_NEAR(val[1], s_test * s_test, 1e-13, "Curve eval Y");
    EXPECT_NEAR(val[2], s_test * s_test * s_test, 1e-13, "Curve eval Z");

    EXPECT_NEAR(d1[0], 1.0, 1e-12, "Curve 1st deriv X");
    EXPECT_NEAR(d1[1], 2.0 * s_test, 1e-12, "Curve 1st deriv Y");
    EXPECT_NEAR(d1[2], 3.0 * s_test * s_test, 1e-12, "Curve 1st deriv Z");

    EXPECT_NEAR(d2[0], 0.0, 1e-11, "Curve 2nd deriv X");
    EXPECT_NEAR(d2[1], 2.0, 1e-11, "Curve 2nd deriv Y");
    EXPECT_NEAR(d2[2], 6.0 * s_test, 1e-11, "Curve 2nd deriv Z");
    std::cout << "[PASS] Test 1: Curve Interpolation, Evaluation & 1st/2nd Derivatives.\n";

    // ------------------------------------------------------------------------
    // 测试 2：节点插入 (Knot Insertion) 几何不变性测试
    // ------------------------------------------------------------------------
    const Real len_before = crv.calc_length();
    const int inserted = crv.insert_knot(0.42, 2);
    const Real len_after = crv.calc_length();
    VecX val_after_insert = crv.eval_point(s_test);

    EXPECT_NEAR(inserted, 2, 0, "Knot insertion count");
    EXPECT_NEAR(len_before, len_after, 1e-13, "Arc length invariance after knot insertion");
    EXPECT_NEAR((val - val_after_insert).norm(), 0.0, 1e-13, "Point invariance after knot insertion");
    std::cout << "[PASS] Test 2: Boehm Knot Insertion & Gauss-Legendre Arc Length.\n";

    // ------------------------------------------------------------------------
    // 测试 3：多线程批量点集投影 (Point Projection)
    // ------------------------------------------------------------------------
    const int n_proj = 2000;
    VecX s_exact = VecX::LinSpaced(n_proj, 0.05, 0.95);
    MatX curve_pts = crv.eval_points_batch(s_exact);

    // 执行投影：曲线上的点投影回自身，距离应为 0，恢复出的参数 s 应等于 s_exact
    auto [s_proj, diff_proj] = crv.project_points(curve_pts, 30, 1e-12);
    EXPECT_NEAR((s_proj - s_exact).lpNorm<Eigen::Infinity>(), 0.0, 1e-9, "Batch point projection parameter s");
    EXPECT_NEAR(diff_proj.lpNorm<Eigen::Infinity>(), 0.0, 1e-10, "Batch point projection distance");
    std::cout << "[PASS] Test 3: Multithreaded Batch Point Projection (2000 points).\n";

    // ------------------------------------------------------------------------
    // 测试 4：两条空间异面直线样条的最近距离投影 (Curve-Curve Projection)
    // ------------------------------------------------------------------------
    MatX line1_pts(2, 3);
    line1_pts << 0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0; // 沿 X 轴: (s, 0, 0)
    MatX line2_pts(2, 3);
    line2_pts << 0.4, -0.5, 2.5,
                 0.4,  0.5, 2.5; // 沿 Y 轴且高度 Z=2.5: (0.4, t - 0.5, 2.5)

    auto [line1, s1] = Curve::fit_from_points(line1_pts, 2);
    auto [line2, s2] = Curve::fit_from_points(line2_pts, 2);

    auto [opt_s, opt_t, min_diff] = line1.project_curve(line2);
    EXPECT_NEAR(opt_s, 0.4, 1e-10, "Curve-Curve closest s");
    EXPECT_NEAR(opt_t, 0.5, 1e-10, "Curve-Curve closest t");
    EXPECT_NEAR(min_diff.norm(), 2.5, 1e-10, "Curve-Curve minimum distance");
    std::cout << "[PASS] Test 4: Curve-to-Curve Closest Distance Projection.\n";

    std::cout << "========================================================\n";
    std::cout << "ALL CURVE MODULE TESTS PASSED SUCCESSFULLY!\n";
    std::cout << "========================================================\n";
    return 0;
}
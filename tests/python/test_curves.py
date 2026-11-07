import sys
import os
import time
import numpy as np

# 将 python/ 目录加入模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../python")))

import fastgeo

def test_curve_bindings():
    print("========================================================")
    print("Running FastGeo Python Bindings Test (nanobind + Eigen)...")
    print(f"Available OpenMP Threads: {fastgeo.get_max_threads()}")
    print("========================================================")

    # 1. 测试曲线精确插值与求值/导数
    t = np.linspace(0.0, 1.0, 25)
    pts = np.column_stack([t, t**2, t**3])  # (25, 3) C-contiguous float64

    crv, s_param = fastgeo.Curve.fit_from_points(pts, k=4, s=t)
    assert crv.k == 4 and crv.num_ctl == 25 and crv.ndim == 3

    s_test = 0.65
    val = crv.eval_point(s_test)
    d1 = crv.eval_deriv(s_test)
    d2 = crv.eval_second_deriv(s_test)

    np.testing.assert_allclose(val, [s_test, s_test**2, s_test**3], atol=1e-13)
    np.testing.assert_allclose(d1, [1.0, 2 * s_test, 3 * s_test**2], atol=1e-12)
    np.testing.assert_allclose(d2, [0.0, 2.0, 6 * s_test], atol=1e-11)
    print("[PASS] 1. Python -> C++ Curve Fit, Eval & Derivatives.")

    # 2. 测试零拷贝原地批量求值 (eval_points_inplace)
    n_batch = 100_000
    s_batch = np.linspace(0.0, 1.0, n_batch)
    out_buf = np.empty((n_batch, 3), dtype=np.float64)
    ptr_before = out_buf.__array_interface__["data"][0]

    crv.eval_points_inplace(s_batch, out_buf)
    ptr_after = out_buf.__array_interface__["data"][0]

    assert ptr_before == ptr_after, "Memory address changed! Not zero-copy!"
    np.testing.assert_allclose(out_buf[:, 0], s_batch, atol=1e-13)
    np.testing.assert_allclose(out_buf[:, 1], s_batch**2, atol=1e-13)
    print(f"[PASS] 2. Zero-Copy Inplace Batch Evaluation ({n_batch} points).")

    # 3. 测试最小二乘拟合 (LMS) 与节点插入
    crv_lms, _ = fastgeo.Curve.fit_from_points(pts, k=4, n_ctl=10, s=t)
    assert crv_lms.num_ctl == 10
    len_before = crv_lms.calc_length()
    inserted = crv_lms.insert_knot(0.5, 2)
    len_after = crv_lms.calc_length()
    assert inserted == 2 and crv_lms.num_ctl == 12
    np.testing.assert_allclose(len_before, len_after, atol=1e-13)
    print("[PASS] 3. LMS Fitting, Knot Insertion & Arc Length.")

    # 4. 测试多线程点集逆向投影与加速性能
    n_proj = 20_000
    s_true = np.linspace(0.05, 0.95, n_proj)
    target_pts = crv.eval_points(s_true)

    t0 = time.perf_counter()
    s_proj, diff = crv.project_points(target_pts, max_iter=25, tol=1e-12)
    dt = (time.perf_counter() - t0) * 1000.0

    np.testing.assert_allclose(s_proj, s_true, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(diff, axis=1), 0.0, atol=1e-10)
    print(f"[PASS] 4. Multithreaded Point Projection ({n_proj} pts in {dt:.2f} ms).")

    print("========================================================")
    print("ALL PYTHON BINDING TESTS PASSED SUCCESSFULLY!")
    print("========================================================")

if __name__ == "__main__":
    test_curve_bindings()
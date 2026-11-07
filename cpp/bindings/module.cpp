// Placeholder
#include <nanobind/nanobind.h>
#include "fastgeo/core/parallel.hpp"

namespace nb = nanobind;

// 声明子模块绑定函数
void bind_spline(nb::module_& m);
void bind_geo(nb::module_& m);

NB_MODULE(_fastgeo_core, m) {
    m.doc() = "FastGeo: High-performance C++20/OpenMP backend for B-spline, FFD and geometric constraints";

    // 全局 OpenMP 线程控制接口
    m.def("set_num_threads", &fastgeo::parallel::set_num_threads,
          nb::arg("num_threads"),
          "Set the number of OpenMP threads used by FastGeo.");
    m.def("get_max_threads", &fastgeo::parallel::get_max_threads,
          "Get the maximum number of OpenMP threads currently available.");

    // 绑定 spline 与 geo 子模块内容
    bind_spline(m);
    bind_geo(m);
}
cmake_minimum_required(VERSION 3.26.0)
project(fastgeo LANGUAGES CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 静态库链接到 Python 扩展模块必需

# 【关键修复 1】生成 compile_commands.json，供 VS Code IntelliSense 消除头文件红线
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 选项控制：允许只编译 C++ 测试而不必每次构建 Python 绑定
option(FASTGEO_BUILD_PYTHON "Build Python bindings via nanobind" ON)
option(FASTGEO_BUILD_CPP_TESTS "Build pure C++ unit tests" ON)

# 1. 查找依赖：Eigen3 与 OpenMP
include(FetchContent)

# 【关键修复 2】Eigen 是纯头文件库，使用 SOURCE_SUBDIR 跳过其内部缓慢的 CMake 检测
FetchContent_Declare(
    Eigen3
    URL "${CMAKE_CURRENT_SOURCE_DIR}/externals/eigen-3.4.0.zip"
    URL_HASH SHA256=EBA3F3D414D2F8CBA2919C78EC6DAAB08FC71BA2BA4AE502B7E5D4D99FC02CDA
    SOURCE_SUBDIR "cmake_dummy_skip" # 故意指定不存在的子目录，仅解压不执行其 CMakeLists.txt
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Eigen3)

# 手动定义干净的 Eigen3::Eigen 纯头文件接口目标
if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3_Eigen INTERFACE)
    target_include_directories(Eigen3_Eigen SYSTEM INTERFACE "${eigen3_SOURCE_DIR}")
    add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)
endif()

find_package(OpenMP REQUIRED)

# 【关键修复 3】如果源文件尚未创建，自动生成空占位文件，防止 CMake Configure 报错中断
set(FASTGEO_CPP_SOURCES
    cpp/src/spline/basis.cpp
    cpp/src/spline/adt.cpp
    cpp/src/spline/curve.cpp
    cpp/src/spline/surface.cpp
    cpp/src/spline/volume.cpp
    cpp/src/geo/dv_geometry.cpp
    cpp/src/geo/constraints.cpp
)
foreach(src_file IN LISTS FASTGEO_CPP_SOURCES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
        get_filename_component(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}" DIRECTORY)
        file(MAKE_DIRECTORY "${src_dir}")
        file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}" "// Placeholder\n")
    endif()
endforeach()

# 2. 构建纯 C++ 核心静态库 (fastgeo_cpp)
add_library(fastgeo_cpp STATIC ${FASTGEO_CPP_SOURCES})

target_include_directories(fastgeo_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
)

# 禁用 Eigen 内部 OpenMP，避免与外层网格点级 OpenMP 产生线程嵌套冲突
target_compile_definitions(fastgeo_cpp PUBLIC EIGEN_DONT_PARALLELIZE)

target_link_libraries(fastgeo_cpp PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# 跨平台优化编译选项 (兼容 Linux GCC/Clang, Windows MinGW-w64 及 MSVC)
if(MSVC)
    target_compile_options(fastgeo_cpp PRIVATE /O2 /fp:fast /bigobj)
else()
    target_compile_options(fastgeo_cpp PRIVATE -O3 -ffast-math -Wall -Wextra)
endif()

# 3. 构建 nanobind Python 扩展模块 (_fastgeo_core)
if(FASTGEO_BUILD_PYTHON)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

    FetchContent_Declare(
        nanobind
        URL "${CMAKE_CURRENT_SOURCE_DIR}/externals/nanobind.zip"
        URL_HASH SHA256=AF87F5EBB5386CA6D09B9A525AF18FC080BBF9C61BF5B0480B3EB17A9C4B8A8E
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nanobind)

    set(FASTGEO_BINDING_SOURCES
        cpp/bindings/module.cpp
        cpp/bindings/bind_spline.cpp
        cpp/bindings/bind_geo.cpp
    )
    foreach(bind_file IN LISTS FASTGEO_BINDING_SOURCES)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${bind_file}")
            get_filename_component(bind_dir "${CMAKE_CURRENT_SOURCE_DIR}/${bind_file}" DIRECTORY)
            file(MAKE_DIRECTORY "${bind_dir}")
            file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/${bind_file}" "// Placeholder\n")
        endif()
    endforeach()

    nanobind_add_module(_fastgeo_core
        STABLE_ABI
        NB_STATIC
        ${FASTGEO_BINDING_SOURCES}
    )

    target_link_libraries(_fastgeo_core PRIVATE fastgeo_cpp)

# 【Windows MinGW 专属优化】静态链接 libgcc 和 libstdc++，避免缺失标准库 DLL
    if(MINGW)
        target_link_options(_fastgeo_core PRIVATE -static-libgcc -static-libstdc++)
    endif()

    set_target_properties(_fastgeo_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/fastgeo"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/fastgeo"
    )
    
    install(TARGETS _fastgeo_core LIBRARY DESTINATION fastgeo)
endif()




#   test 

# 4. 构建纯 C++ 单元测试可执行文件
if(FASTGEO_BUILD_CPP_TESTS)
    enable_testing()
    add_executable(test_spline_core "./tests/cpp/test_spline_core.cpp")
    target_link_libraries(test_spline_core PRIVATE fastgeo_cpp)
    add_test(NAME test_spline_core COMMAND test_spline_core)
endif()
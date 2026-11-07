#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>

namespace fastgeo {

// 基础标量与索引类型
using Real = double;
using Index = std::ptrdiff_t; // 有符号索引，兼容 OpenMP 循环与 Eigen

// 固定大小几何向量与矩阵 (用于 3D 空间物理坐标、参数坐标 (u,v,w) 与局部雅可比求逆)
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;

using Mat2 = Eigen::Matrix<Real, 2, 2, Eigen::RowMajor>;
using Mat3 = Eigen::Matrix<Real, 3, 3, Eigen::RowMajor>;

// 动态一维向量 (用于节点向量 knots、参数数组 u/v/w 等)
using VecX = Eigen::VectorXd;
using VecXi = Eigen::VectorXi;

// 动态二维行优先矩阵 (强制 RowMajor，与 NumPy C_CONTIGUOUS 内存布局 100% 一致)
using MatX = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatX3 = Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>; // N x 3 网格点坐标阵
using MatX2 = Eigen::Matrix<Real, Eigen::Dynamic, 2, Eigen::RowMajor>; // N x 2 曲面参数坐标阵

// 零拷贝内存映射视图 (直接包装外部裸指针或 nanobind::ndarray)
using MapVec3 = Eigen::Map<Vec3>;
using ConstMapVec3 = Eigen::Map<const Vec3>;

using MapVecX = Eigen::Map<VecX>;
using ConstMapVecX = Eigen::Map<const VecX>;

using MapMatX = Eigen::Map<MatX>;
using ConstMapMatX = Eigen::Map<const MatX>;

using MapMatX3 = Eigen::Map<MatX3>;
using ConstMapMatX3 = Eigen::Map<const MatX3>;

// 稀疏矩阵类型 (采用 RowMajor 即 CSR 格式，方便直接零拷贝导出为 scipy.sparse.csr_matrix)
using SparseMat = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<Real>;

using ColMatX = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
} // namespace fastgeo
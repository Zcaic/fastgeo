#pragma once

#include "fastgeo/core/types.hpp"
#include <cassert>
#include <type_traits>
#include <vector>

namespace fastgeo {

// ============================================================================
// 3D 数组零开销视图 (Row-Major: n0 x n1 x n2, 对应 Surface 控制点 nu x nv x ndim)
// ============================================================================
template <typename T>
class Tensor3DView {
public:
    using value_type = std::remove_cv_t<T>;
    using pointer = T*;
    using reference = T&;

    constexpr Tensor3DView() noexcept : data_(nullptr), n0_(0), n1_(0), n2_(0) {}

    constexpr Tensor3DView(T* data, size_t n0, size_t n1, size_t n2) noexcept
        : data_(data), n0_(n0), n1_(n1), n2_(n2) {}

    // 允许从非 const 视图隐式转换为 const 视图
    template <typename U, typename = std::enable_if_t<std::is_same_v<T, const U>>>
    constexpr Tensor3DView(const Tensor3DView<U>& other) noexcept
        : data_(other.data()), n0_(other.shape(0)), n1_(other.shape(1)), n2_(other.shape(2)) {}

    // 标量元素访问: view(i, j, d)
    inline reference operator()(size_t i, size_t j, size_t d) const noexcept {
        assert(i < n0_ && j < n1_ && d < n2_);
        return data_[(i * n1_ + j) * n2_ + d];
    }

    // 获取 (i, j) 处控制点的首地址
    inline pointer ptr_at(size_t i, size_t j) const noexcept {
        assert(i < n0_ && j < n1_);
        return &data_[(i * n1_ + j) * n2_];
    }

    // 直接映射为 3D 空间向量 (当 n2 == 3 时极度高频使用)
    inline auto vec3(size_t i, size_t j) const noexcept {
        assert(n2_ == 3);
        using VecType = std::conditional_t<std::is_const_v<T>, const Vec3, Vec3>;
        return Eigen::Map<VecType>(ptr_at(i, j));
    }

    // 映射为任意长度 N (n2) 的向量 (兼容 ndim != 3 的广义样条)
    inline auto vec(size_t i, size_t j) const noexcept {
        using VecType = std::conditional_t<std::is_const_v<T>, const VecX, VecX>;
        return Eigen::Map<VecType>(ptr_at(i, j), static_cast<Index>(n2_));
    }

    constexpr T* data() const noexcept { return data_; }
    constexpr size_t size() const noexcept { return n0_ * n1_ * n2_; }
    constexpr size_t shape(size_t dim) const noexcept {
        return dim == 0 ? n0_ : (dim == 1 ? n1_ : n2_);
    }

private:
    T* data_;
    size_t n0_, n1_, n2_;
};

// ============================================================================
// 4D 数组零开销视图 (Row-Major: n0 x n1 x n2 x n3, 对应 Volume 控制点 nu x nv x nw x ndim)
// ============================================================================
template <typename T>
class Tensor4DView {
public:
    using value_type = std::remove_cv_t<T>;
    using pointer = T*;
    using reference = T&;

    constexpr Tensor4DView() noexcept : data_(nullptr), n0_(0), n1_(0), n2_(0), n3_(0) {}

    constexpr Tensor4DView(T* data, size_t n0, size_t n1, size_t n2, size_t n3) noexcept
        : data_(data), n0_(n0), n1_(n1), n2_(n2), n3_(n3) {}

    // 允许从非 const 视图隐式转换为 const 视图
    template <typename U, typename = std::enable_if_t<std::is_same_v<T, const U>>>
    constexpr Tensor4DView(const Tensor4DView<U>& other) noexcept
        : data_(other.data()), n0_(other.shape(0)), n1_(other.shape(1)),
          n2_(other.shape(2)), n3_(other.shape(3)) {}

    // 标量元素访问: view(i, j, k, d)
    inline reference operator()(size_t i, size_t j, size_t k, size_t d) const noexcept {
        assert(i < n0_ && j < n1_ && k < n2_ && d < n3_);
        return data_[((i * n1_ + j) * n2_ + k) * n3_ + d];
    }

    // 获取 (i, j, k) 处控制点的首地址
    inline pointer ptr_at(size_t i, size_t j, size_t k) const noexcept {
        assert(i < n0_ && j < n1_ && k < n2_);
        return &data_[((i * n1_ + j) * n2_ + k) * n3_];
    }

    // 直接映射为 3D 空间向量 (FFD 控制点求值与变形核心接口)
    inline auto vec3(size_t i, size_t j, size_t k) const noexcept {
        assert(n3_ == 3);
        using VecType = std::conditional_t<std::is_const_v<T>, const Vec3, Vec3>;
        return Eigen::Map<VecType>(ptr_at(i, j, k));
    }

    // 映射为任意长度 N (n3) 的向量
    inline auto vec(size_t i, size_t j, size_t k) const noexcept {
        using VecType = std::conditional_t<std::is_const_v<T>, const VecX, VecX>;
        return Eigen::Map<VecType>(ptr_at(i, j, k), static_cast<Index>(n3_));
    }

    constexpr T* data() const noexcept { return data_; }
    constexpr size_t size() const noexcept { return n0_ * n1_ * n2_ * n3_; }
    constexpr size_t shape(size_t dim) const noexcept {
        return dim == 0 ? n0_ : (dim == 1 ? n1_ : (dim == 2 ? n2_ : n3_));
    }

private:
    T* data_;
    size_t n0_, n1_, n2_, n3_;
};

// ============================================================================
// 拥有内存所有权的 3D / 4D 容器 (用于 C++ 内部类成员持久化存储控制点)
// ============================================================================
class Tensor3D {
public:
    Tensor3D() = default;
    Tensor3D(size_t n0, size_t n1, size_t n2, Real init_val = 0.0)
        : n0_(n0), n1_(n1), n2_(n2), storage_(n0 * n1 * n2, init_val) {}

    void resize(size_t n0, size_t n1, size_t n2, Real init_val = 0.0) {
        n0_ = n0; n1_ = n1; n2_ = n2;
        storage_.assign(n0 * n1 * n2, init_val);
    }

    Tensor3DView<Real> view() noexcept { return {storage_.data(), n0_, n1_, n2_}; }
    Tensor3DView<const Real> view() const noexcept { return {storage_.data(), n0_, n1_, n2_}; }

    operator Tensor3DView<Real>() noexcept { return view(); }
    operator Tensor3DView<const Real>() const noexcept { return view(); }

    inline Real& operator()(size_t i, size_t j, size_t d) noexcept { return view()(i, j, d); }
    inline const Real& operator()(size_t i, size_t j, size_t d) const noexcept { return view()(i, j, d); }

    inline auto vec3(size_t i, size_t j) noexcept { return view().vec3(i, j); }
    inline auto vec3(size_t i, size_t j) const noexcept { return view().vec3(i, j); }

    Real* data() noexcept { return storage_.data(); }
    const Real* data() const noexcept { return storage_.data(); }
    size_t shape(size_t dim) const noexcept { return view().shape(dim); }
    size_t size() const noexcept { return storage_.size(); }

private:
    size_t n0_ = 0, n1_ = 0, n2_ = 0;
    std::vector<Real> storage_;
};

class Tensor4D {
public:
    Tensor4D() = default;
    Tensor4D(size_t n0, size_t n1, size_t n2, size_t n3, Real init_val = 0.0)
        : n0_(n0), n1_(n1), n2_(n2), n3_(n3), storage_(n0 * n1 * n2 * n3, init_val) {}

    void resize(size_t n0, size_t n1, size_t n2, size_t n3, Real init_val = 0.0) {
        n0_ = n0; n1_ = n1; n2_ = n2; n3_ = n3;
        storage_.assign(n0 * n1 * n2 * n3, init_val);
    }

    Tensor4DView<Real> view() noexcept { return {storage_.data(), n0_, n1_, n2_, n3_}; }
    Tensor4DView<const Real> view() const noexcept { return {storage_.data(), n0_, n1_, n2_, n3_}; }

    operator Tensor4DView<Real>() noexcept { return view(); }
    operator Tensor4DView<const Real>() const noexcept { return view(); }

    inline Real& operator()(size_t i, size_t j, size_t k, size_t d) noexcept { return view()(i, j, k, d); }
    inline const Real& operator()(size_t i, size_t j, size_t k, size_t d) const noexcept { return view()(i, j, k, d); }

    inline auto vec3(size_t i, size_t j, size_t k) noexcept { return view().vec3(i, j, k); }
    inline auto vec3(size_t i, size_t j, size_t k) const noexcept { return view().vec3(i, j, k); }

    Real* data() noexcept { return storage_.data(); }
    const Real* data() const noexcept { return storage_.data(); }
    size_t shape(size_t dim) const noexcept { return view().shape(dim); }
    size_t size() const noexcept { return storage_.size(); }

private:
    size_t n0_ = 0, n1_ = 0, n2_ = 0, n3_ = 0;
    std::vector<Real> storage_;
};

} // namespace fastgeo
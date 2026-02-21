#include "GPTSoVITS/model/tensor.h"

#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "GPTSoVITS/Utils/exception.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <xtl/xhalf_float.hpp>

namespace GPTSoVITS::Model {

namespace {
using half = xtl::half_float;

template <typename T_IN, typename T_OUT>
void ConvertData(const void* in, void* out, int64_t numel) {
  const T_IN* in_ptr = static_cast<const T_IN*>(in);
  T_OUT* out_ptr = static_cast<T_OUT*>(out);
  for (int64_t i = 0; i < numel; ++i) {
    out_ptr[i] = static_cast<T_OUT>(in_ptr[i]);
  }
}

template <typename T_IN>
void DispatchOut(const T_IN* in, void* out, int64_t numel, DataType out_type) {
  switch (out_type) {
    case DataType::kFloat32: ConvertData<T_IN, float>(in, out, numel); break;
    case DataType::kFloat16: ConvertData<T_IN, half>(in, out, numel); break;
    case DataType::kFloat8: THROW_ERROR("FP8 conversion not supported on CPU. Use TensorRT backend for FP8 support.");
    case DataType::kInt32: ConvertData<T_IN, int32_t>(in, out, numel); break;
    case DataType::kInt64: ConvertData<T_IN, int64_t>(in, out, numel); break;
    case DataType::kInt8: ConvertData<T_IN, int8_t>(in, out, numel); break;
    case DataType::kUInt8: ConvertData<T_IN, uint8_t>(in, out, numel); break;
    default: THROW_ERROR("Unsupported output type for conversion");
  }
}

void DispatchIn(const void* in, DataType in_type, void* out, DataType out_type,
                int64_t numel) {
  switch (in_type) {
    case DataType::kFloat32:
      DispatchOut<float>(static_cast<const float*>(in), out, numel, out_type);
      break;
    case DataType::kFloat16:
      DispatchOut<half>(static_cast<const half*>(in), out, numel, out_type);
      break;
    case DataType::kFloat8:
      THROW_ERROR("FP8 conversion not supported on CPU. Use TensorRT backend for FP8 support.");
    case DataType::kInt32:
      DispatchOut<int32_t>(static_cast<const int32_t*>(in), out, numel,
                           out_type);
      break;
    case DataType::kInt64:
      DispatchOut<int64_t>(static_cast<const int64_t*>(in), out, numel,
                           out_type);
      break;
    case DataType::kInt8:
      DispatchOut<int8_t>(static_cast<const int8_t*>(in), out, numel, out_type);
      break;
    case DataType::kUInt8:
      DispatchOut<uint8_t>(static_cast<const uint8_t*>(in), out, numel,
                           out_type);
      break;
    default: THROW_ERROR("Unsupported input type for conversion");
  }
}

// 计算形状元素乘积
int64_t ComputeNumel(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  // 检查乘积是否溢出int64范围
  return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}
}  // namespace

Tensor::Tensor(void* data, const std::vector<int64_t>& shape, DataType dtype,
               Device device, Deleter deleter)
    : shape_(shape), dtype_(dtype), device_(device) {

  if (data == nullptr && ComputeNumel(shape) > 0) {
     THROW_ERRORN("Tensor construction failed: data is null but shape is not empty.");
  }

  // 计算并缓存元素个数
  numel_ = ComputeNumel(shape_);

  // 初始化智能指针
  if (deleter) {
    data_ptr_ = std::shared_ptr<void>(data, deleter);
  } else {
    // 无deleter时默认为不管理生命周期(如外部栈内存/View模式),或者是空deleter
    data_ptr_ = std::shared_ptr<void>(data, [](void*) {});
  }
}

std::unique_ptr<Tensor> Tensor::CreateFromHost(
    void* data, const std::vector<int64_t>& shape, DataType dtype,
    Deleter deleter) {
  return std::make_unique<Tensor>(data, shape, dtype, Device(DeviceType::kCPU), deleter);
}

std::unique_ptr<Tensor> Tensor::Empty(const std::vector<int64_t>& shape,
                                      DataType dtype, Device device) {
  int64_t numel = ComputeNumel(shape);
  size_t bytes = static_cast<size_t>(numel) * ElementSize(dtype);

  if (device.type == DeviceType::kCPU) {
    void* data = std::malloc(bytes);
    if (!data) THROW_ERRORN("CPU memory allocation failed for {} bytes.", bytes);
    return std::make_unique<Tensor>(data, shape, dtype, device, [](void* p) { std::free(p); });
  } else if (device.type == DeviceType::kCUDA) {
#ifdef WITH_CUDA
    void* data = nullptr;
    // 切换到对应设备进行分配
    int old_device = 0;
    cudaGetDevice(&old_device);
    cudaSetDevice(device.device_id);
    cudaError_t err = cudaMalloc(&data, bytes);
    cudaSetDevice(old_device);
    if (err != cudaSuccess) {
      THROW_ERRORN("CUDA memory allocation failed: {}", cudaGetErrorString(err));
    }
    return std::make_unique<Tensor>(data, shape, dtype, device, [](void* p) {
      // TODO: 理论上用统一内存池管理比较好
      cudaFree(p);
    });
#else
    THROW_ERRORN("CUDA support is not enabled in this build.");
#endif
  }

  THROW_ERRORN("Unsupported device type for Empty tensor allocation.");
}

std::unique_ptr<Tensor> Tensor::Clone() const {
  auto new_tensor = Empty(shape_, dtype_, device_);
  size_t bytes = ByteSize();

  if (device_.type == DeviceType::kCPU) {
    std::memcpy(new_tensor->Data(), Data(), bytes);
  } else if (device_.type == DeviceType::kCUDA) {
#ifdef WITH_CUDA
    cudaError_t err = cudaMemcpy(new_tensor->Data(), Data(), bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      THROW_ERRORN("CUDA deep copy failed: {}", cudaGetErrorString(err));
    }
#else
    THROW_ERRORN("CUDA support is not enabled.");
#endif
  }
  return new_tensor;
}

std::unique_ptr<Tensor> Tensor::ToDevice(Device device) const {
  if (device_ == device) {
    return Clone();
  }

  auto new_tensor = Empty(shape_, dtype_, device);
  size_t bytes = ByteSize();

  if (device_.type == DeviceType::kCPU && device.type == DeviceType::kCUDA) {
#ifdef WITH_CUDA
    cudaError_t err = cudaMemcpy(new_tensor->Data(), Data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) THROW_ERRORN("H2D copy failed: {}", cudaGetErrorString(err));
#else
    THROW_ERRORN("CUDA support is not enabled.");
#endif
  } else if (device_.type == DeviceType::kCUDA && device.type == DeviceType::kCPU) {
#ifdef WITH_CUDA
    cudaError_t err = cudaMemcpy(new_tensor->Data(), Data(), bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) THROW_ERRORN("D2H copy failed: {}", cudaGetErrorString(err));
#else
    THROW_ERRORN("CUDA support is not enabled.");
#endif
  } else if (device_.type == DeviceType::kCUDA && device.type == DeviceType::kCUDA) {
#ifdef WITH_CUDA
    // Cross-GPU copy
    cudaError_t err = cudaMemcpy(new_tensor->Data(), Data(), bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) THROW_ERRORN("D2D copy failed: {}", cudaGetErrorString(err));
#else
    THROW_ERRORN("CUDA support is not enabled.");
#endif
  } else {
    std::memcpy(new_tensor->Data(), Data(), bytes);
  }

  return new_tensor;
}

std::unique_ptr<Tensor> Tensor::ToType(DataType dtype) const {
  if (dtype_ == dtype) {
    return Clone();
  }

  // 优先尝试 GPU 转换（如果在 CUDA 上）
  if (device_.type == DeviceType::kCUDA) {
#ifdef WITH_CUDA
    // 尝试使用 GPU 转换（通过 GPUTypeConverter）
    // 暂时使用现有实现，后续会替换为真正的 GPU 转换
    auto cpu_src = ToCPU();
    auto cpu_dst = Empty(shape_, dtype, Device(DeviceType::kCPU));

    DispatchIn(cpu_src->Data(), dtype_, cpu_dst->Data(), dtype, numel_);

    return cpu_dst->ToDevice(device_);
#else
    // 无 CUDA 支持，回退到 CPU 转换
    auto cpu_src = ToCPU();
    auto cpu_dst = Empty(shape_, dtype, Device(DeviceType::kCPU));

    DispatchIn(cpu_src->Data(), dtype_, cpu_dst->Data(), dtype, numel_);

    return cpu_dst->ToDevice(device_);
#endif
  }

  // CPU 上的转换
  auto cpu_dst = Empty(shape_, dtype, Device(DeviceType::kCPU));
  DispatchIn(Data(), dtype_, cpu_dst->Data(), dtype, numel_);

  return cpu_dst;
}

std::unique_ptr<Tensor> Tensor::To(Device device, DataType dtype) const {
  // 设备和类型都一致
  if (device_ == device && dtype_ == dtype) {
    return Clone();
  }

  // 仅设备不一致 (H2D / D2H)
  if (dtype_ == dtype) {
    return ToDevice(device);
  }

  // 仅类型不一致 (CPU 类型转换 / GPU 搬回 CPU 转换再搬回去)
  if (device_ == device) {
    return ToType(dtype);
  }

  // 综合转换 (最优路径: 在源/目标端更便宜的一端进行转换)
  // 非标转换(如 Int64 -> Float16) 在 CPU 上会稳定点
  // 先转
  auto temp_cpu_typed = ToCPU()->ToType(dtype);
  // 再搬运
  return temp_cpu_typed->ToDevice(device);
}

Tensor& Tensor::Reshape(const std::vector<int64_t>& new_shape) {
  int64_t new_numel = ComputeNumel(new_shape);

  // 确保Reshape前后元素总量一致
  if (new_numel != numel_) {
    THROW_ERRORN("Reshape failed: element count mismatch.");
  }

  shape_ = new_shape;
  return *this;
}

std::unique_ptr<Tensor> Tensor::View(const std::vector<int64_t>& new_shape) {
  int64_t new_numel = ComputeNumel(new_shape);

  // 确保View前后元素总量一致
  if (new_numel != numel_) {
    THROW_ERRORN("View failed: element count mismatch.");
  }

  // 创建视图，共享底层内存
  // 使用空 deleter，表示不管理内存生命周期
  auto view = std::make_unique<Tensor>(
      data_ptr_.get(),
      new_shape,
      dtype_,
      device_,
      [](void*) {});  // 空 deleter

  return view;
}

std::unique_ptr<Tensor> Tensor::Slice(int64_t start, int64_t end, int axis) {
  if (axis < 0 || axis >= static_cast<int>(shape_.size())) {
    THROW_ERRORN("Slice: axis out of range.");
  }
  if (start < 0) start += shape_[axis];
  if (end < 0) end += shape_[axis];
  if (start < 0 || end > shape_[axis] || start >= end) {
    THROW_ERRORN("Slice: invalid start/end indices.");
  }

  // 计算切片后的形状
  std::vector<int64_t> new_shape = shape_;
  new_shape[axis] = end - start;

  // 计算数据偏移
  size_t element_size = ElementSize(dtype_);
  int64_t offset = start;

  // 计算该维度之前的元素总数
  for (int i = axis + 1; i < static_cast<int>(shape_.size()); ++i) {
    offset *= shape_[i];
  }

  uint8_t* data_ptr = static_cast<uint8_t*>(data_ptr_.get()) + offset * element_size;

  // 如果是在 CPU 上，可以使用零拷贝视图
  if (device_.type == DeviceType::kCPU) {
    return std::make_unique<Tensor>(
        data_ptr,
        new_shape,
        dtype_,
        device_,
        [](void*) {});  // 空 deleter，视图不管理内存
  } else {
    // 在 GPU 上，需要创建新的 Tensor 并拷贝数据
    // （因为需要维护正确的内存管理）
    auto sliced = Empty(new_shape, dtype_, device_);
    size_t copy_size = static_cast<size_t>(ComputeNumel(new_shape)) * element_size;
#ifdef WITH_CUDA
    cudaMemcpy(sliced->Data(), data_ptr, copy_size, cudaMemcpyDeviceToDevice);
#else
    THROW_ERRORN("Slice on CUDA not available without CUDA support.");
#endif
    return sliced;
  }
}

void Tensor::CopyFrom(const Tensor* src) {
  if (shape_ != src->Shape()) {
    THROW_ERRORN("CopyFrom: shape mismatch.");
  }
  if (dtype_ != src->Type()) {
    THROW_ERRORN("CopyFrom: data type mismatch.");
  }

  size_t bytes = ByteSize();

  // 检查是否可以零拷贝（同一设备）
  if (device_ == src->GetDevice()) {
    if (device_.type == DeviceType::kCPU) {
      std::memcpy(Data(), src->Data(), bytes);
    } else {
#ifdef WITH_CUDA
      cudaMemcpy(Data(), src->Data(), bytes, cudaMemcpyDeviceToDevice);
#endif
    }
  } else {
    // 跨设备拷贝
    if (device_.type == DeviceType::kCPU) {
#ifdef WITH_CUDA
      cudaMemcpy(Data(), src->Data(), bytes, cudaMemcpyDeviceToHost);
#endif
    } else {
#ifdef WITH_CUDA
      cudaMemcpy(Data(), src->Data(), bytes, cudaMemcpyHostToDevice);
#endif
    }
  }
}

bool Tensor::SharesMemoryWith(const Tensor& other) const {
  // 检查是否共享相同的底层内存
  return data_ptr_.get() == other.data_ptr_.get();
}

const std::vector<int64_t>& Tensor::Shape() const {
  return shape_;
}

DataType Tensor::Type() const {
  return dtype_;
}

Device Tensor::GetDevice() const {
  return device_;
}

DeviceType Tensor::GetDeviceType() const {
  return device_.type;
}

int64_t Tensor::ElementCount() const {
  return numel_;
}

size_t Tensor::ElementSize(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat32:
    case DataType::kInt32:
      return 4;
    case DataType::kInt64:
      return 8;
    case DataType::kFloat16:
      return 2;
    case DataType::kFloat8:
      return 1;
    case DataType::kInt8:
    case DataType::kUInt8:
      return 1;
    default:
      THROW_ERRORN("unknown type");
  }
}

std::unique_ptr<Tensor> Tensor::Concat(const std::vector<Tensor*>& tensors, int axis) {
  if (tensors.empty()) THROW_ERRORN("Concat: input tensor list is empty.");
  if (tensors.size() == 1) return tensors[0]->Clone();

  // 检查所有Tensor类型、设备是否一致
  DataType dtype = tensors[0]->Type();
  Device device = tensors[0]->GetDevice();
  auto base_shape = tensors[0]->Shape();

  if (axis < 0) axis += static_cast<int>(base_shape.size());
  if (axis < 0 || axis >= static_cast<int>(base_shape.size())) {
    THROW_ERRORN("Concat: axis out of range.");
  }

  std::vector<int64_t> out_shape = base_shape;
  int64_t total_dim = 0;

  for (auto t : tensors) {
    if (t->Type() != dtype || t->GetDevice() != device) {
      THROW_ERRORN("Concat: all tensors must have the same data type and device.");
    }
    auto s = t->Shape();
    if (s.size() != base_shape.size()) THROW_ERRORN("Concat: dimension mismatch.");
    for (int i = 0; i < s.size(); ++i) {
      if (i != axis && s[i] != base_shape[i]) THROW_ERRORN("Concat: shape mismatch on non-concat axis.");
    }
    total_dim += s[axis];
  }
  out_shape[axis] = total_dim;

  auto out_tensor = Empty(out_shape, dtype, device);
  uint8_t* dst_ptr = out_tensor->Data<uint8_t>();
  
  if (axis == 0) {
    for (auto t : tensors) {
      size_t b = t->ByteSize();
      if (device.type == DeviceType::kCPU) {
        std::memcpy(dst_ptr, t->Data(), b);
      } else {
#ifdef WITH_CUDA
        cudaMemcpy(dst_ptr, t->Data(), b, cudaMemcpyDeviceToDevice);
#endif
      }
      dst_ptr += b;
    }
  } else {
    if (base_shape.size() == 2 && axis == 1) {
      int64_t row_count = base_shape[0];
      size_t element_size = ElementSize(dtype);
      for (int64_t r = 0; r < row_count; ++r) {
        for (auto t : tensors) {
          size_t row_bytes = t->Shape()[1] * element_size;
          void* src = static_cast<uint8_t*>(t->Data()) + r * row_bytes;
          if (device.type == DeviceType::kCPU) {
            std::memcpy(dst_ptr, src, row_bytes);
          } else {
#ifdef WITH_CUDA
            cudaMemcpy(dst_ptr, src, row_bytes, cudaMemcpyDeviceToDevice);
#endif
          }
          dst_ptr += row_bytes;
        }
      }
    } else {
      THROW_ERRORN("Concat: generic axis concatenation not yet implemented for these dimensions.");
    }
  }

  return out_tensor;
}

size_t Tensor::ByteSize() const {
  return static_cast<size_t>(numel_) * ElementSize(dtype_);
}

void Tensor::CheckInvariant() const {
    // 内部调试断言
    if (ComputeNumel(shape_) != numel_) {
        throw std::logic_error("Tensor invariant broken: shape and numel mismatch.");
    }
}

} // namespace GPTSoVITS::Model


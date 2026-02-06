#include "GPTSoVITS/model/tensor.h"

#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "GPTSoVITS/Utils/exception.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace GPTSoVITS::Model {

namespace {
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

Tensor& Tensor::Reshape(const std::vector<int64_t>& new_shape) {
  int64_t new_numel = ComputeNumel(new_shape);

  // 确保Reshape前后元素总量一致
  if (new_numel != numel_) {
    THROW_ERRORN("Reshape failed: element count mismatch.");
  }

  shape_ = new_shape;
  return *this;
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

  // 仅支持在最高维或当Tensor是连续块时的拼接逻辑
  // 针对 TTS 场景: 
  // 1. PhoneSeq (seq_len) -> axis 0 OK
  // 2. BertSeq (1024, seq_len) -> axis 1
  
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
    // 针对 (1024, seq_len) 在 axis 1 拼接
    // 需要更加复杂的逻辑, 这里针对 2D 场景做特殊优化
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


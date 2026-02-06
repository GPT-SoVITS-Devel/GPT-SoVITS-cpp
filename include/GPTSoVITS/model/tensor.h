#ifndef GPT_SOVITS_CPP_TENSOR_H
#define GPT_SOVITS_CPP_TENSOR_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "GPTSoVITS/model/device.h"

namespace GPTSoVITS::Model {

// 数据类型枚举
enum class DataType {
  kFloat32,
  kFloat16,
  kInt32,
  kInt64,
  kInt8,
  kUInt8
};

class Tensor {
public:
  // 内存释放器回调
  using Deleter = std::function<void(void*)>;

  // 禁止默认构造,必须通过工厂方法或明确参数创建有效Tensor
  Tensor() = delete;

  // 正常情况下,应优先使用Create方法而非直接调用构造函数
  Tensor(void* data, const std::vector<int64_t>& shape, DataType dtype,
         Device device, Deleter deleter = nullptr);

  ~Tensor() = default;

  // 大块内存隐式复制引发性能抖动
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  // 允许移动语义
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;

  // 从Host内存创建Tensor(零拷贝/接管所有权)
  static std::unique_ptr<Tensor> CreateFromHost(
      void* data, const std::vector<int64_t>& shape, DataType dtype,
      Deleter deleter = nullptr);

  // 分配新的内存空间
  static std::unique_ptr<Tensor> Empty(const std::vector<int64_t>& shape,
                                       DataType dtype, Device device);

  // 深度拷贝
  [[nodiscard]] std::unique_ptr<Tensor> Clone() const;

  // 设备间拷贝
  [[nodiscard]] std::unique_ptr<Tensor> ToDevice(Device device) const;

  // 类型转换
  [[nodiscard]] std::unique_ptr<Tensor> ToType(DataType dtype) const;

  // 综合转换 (合并设备搬运与类型转换)
  [[nodiscard]] std::unique_ptr<Tensor> To(Device device, DataType dtype) const;

  // 快速移动到CPU
  [[nodiscard]] std::unique_ptr<Tensor> ToCPU() const {
    return ToDevice(Device(DeviceType::kCPU));
  }

  // 仅修改元数据, 不触碰物理内存
  // 返回自身引用以支持链式调用
  Tensor& Reshape(const std::vector<int64_t>& new_shape);

  // 获取底层数据指针
  template <typename T = void>
  T* Data() const {
    return static_cast<T*>(data_ptr_.get());
  }

  // 获取元数据
  const std::vector<int64_t>& Shape() const;
  DataType Type() const;
  Device GetDevice() const;
  DeviceType GetDeviceType() const;

  // 快捷访问 (仅限CPU且类型匹配时安全)
  template <typename T>
  T& At(int64_t index) {
    return static_cast<T*>(data_ptr_.get())[index];
  }

  template <typename T>
  const T& At(int64_t index) const {
    return static_cast<const T*>(data_ptr_.get())[index];
  }

  // 快捷判断
  bool IsCPU() const { return device_.type == DeviceType::kCPU; }
  bool IsCUDA() const { return device_.type == DeviceType::kCUDA; }

  // 计算元素总数
  int64_t ElementCount() const;

  // 计算Tensor占用的字节数
  size_t ByteSize() const;

  /**
   * @brief 在指定维度拼接多个Tensor
   * @param tensors 待拼接的Tensor列表
   * @param axis 拼接维度
   * @return 拼接后的新Tensor
   */
  static std::unique_ptr<Tensor> Concat(const std::vector<Tensor*>& tensors, int axis = 0);

  // 获取单个元素占用的字节数
  static size_t ElementSize(DataType dtype);

private:
  // 检查Shape与总容量是否一致
  void CheckInvariant() const;

  std::shared_ptr<void> data_ptr_;
  std::vector<int64_t> shape_;
  DataType dtype_;
  Device device_;
  int64_t numel_ = 0; // 缓存元素数量,避免重复计算
};

} // namespace GPTSoVITS::Model


#endif // GPT_SOVITS_CPP_TENSOR_H

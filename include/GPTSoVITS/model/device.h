//
// Created by Huiyicc on 2026/1/10.
//

#ifndef GPT_SOVITS_CPP_DEVICE_H
#define GPT_SOVITS_CPP_DEVICE_H

#include <cstdint>

namespace GPTSoVITS::Model {

/**
 * @brief 计算设备类型
 */
enum class DeviceType {
  kCPU = 0,
  kCUDA = 1,      // NVIDIA GPU
  kDirectML = 2,  // Windows DirectML (AMD/Intel/NVIDIA)
  kCoreML = 3     // Apple Silicon
};

/**
 * @brief 设备实例信息封装
 */
struct Device {
  DeviceType type;
  int device_id;
  void* stream;  // 底层计算流句柄 (如 cudaStream_t)

  Device(DeviceType t = DeviceType::kCPU, int id = 0, void* s = nullptr)
      : type(t), device_id(id), stream(s) {}

  bool operator==(const Device& other) const {
    return type == other.type && device_id == other.device_id &&
           stream == other.stream;
  }
  bool operator!=(const Device& other) const {
    return !operator==(other);
  };
};

}  // namespace GPTSoVITS::Model

#endif  // GPT_SOVITS_CPP_DEVICE_H

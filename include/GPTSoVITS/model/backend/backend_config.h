//
// Created by iFlow CLI on 2026/2/20.
//

#ifndef GPT_SOVITS_CPP_BACKEND_CONFIG_H
#define GPT_SOVITS_CPP_BACKEND_CONFIG_H

#include <cstdint>
#include <string>

#include "GPTSoVITS/model/device.h"

namespace GPTSoVITS::Model {

/**
 * @brief 精度模式
 */
enum class PrecisionMode {
  kAuto,     // 自动检测（根据模型输入类型）
  kFP32,     // 强制 FP32
  kFP16,     // 强制 FP16
  kMixed,    // 混合精度（输入自动，计算FP16）
  kINT8      // INT8
};

/**
 * @brief 后端配置结构
 */
struct BackendConfig {
  // 设备配置
  Device device{DeviceType::kCPU, 0};

  // 精度配置
  PrecisionMode precision = PrecisionMode::kAuto;

  // 线程配置
  int work_thread_num = 1;

  // IO Binding 配置
  bool enable_iobinding = true;

  // 内存配置
  // TODO: 未实现，预留，未来扩展
  int64_t max_workspace_size = 0;  // 工作空间大小限制（字节），0表示不限制

  // 验证配置
  void Validate() {
    if (work_thread_num <= 0) {
      work_thread_num = 1;
    }
    if (max_workspace_size < 0) {
      max_workspace_size = 0;
    }
  }

  /**
   * @brief 获取推荐的默认配置
   */
  static BackendConfig GetDefaultConfig() {
    BackendConfig config;
    config.Validate();
    return config;
  }

  /**
   * @brief 获取推理推荐配置
   */
  static BackendConfig GetHighPerformanceConfig() {
    BackendConfig config;
    config.device = Device(DeviceType::kCUDA, 0);
    config.precision = PrecisionMode::kFP16;
    config.work_thread_num = 4;
    config.enable_iobinding = true;
    config.Validate();
    return config;
  }

  /**
   * @brief 获取边缘设备配置
   */
  static BackendConfig GetEdgeConfig() {
    BackendConfig config;
    config.device = Device(DeviceType::kCPU, 0);
    config.precision = PrecisionMode::kFP16;
    config.work_thread_num = 2;
    config.enable_iobinding = true;
    config.max_workspace_size = 512 * 1024 * 1024;  // 512MB
    config.Validate();
    return config;
  }
};

/**
 * @brief 精度模式转字符串（日志）
 */
inline const char* PrecisionModeToString(PrecisionMode mode) {
  switch (mode) {
    case PrecisionMode::kAuto:
      return "Auto";
    case PrecisionMode::kFP32:
      return "FP32";
    case PrecisionMode::kFP16:
      return "FP16";
    case PrecisionMode::kMixed:
      return "Mixed";
    case PrecisionMode::kINT8:
      return "INT8";
    default:
      return "Unknown";
  }
}

}  // namespace GPTSoVITS::Model

#endif  // GPT_SOVITS_CPP_BACKEND_CONFIG_H
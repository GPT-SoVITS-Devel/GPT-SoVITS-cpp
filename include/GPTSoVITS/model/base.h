//
// Created by Huiyicc on 2026/1/10.
//

#ifndef GPT_SOVITS_CPP_MODEL_BASE_H
#define GPT_SOVITS_CPP_MODEL_BASE_H

#include <string>
#include <unordered_map>
#include <vector>

#include "GPTSoVITS/model/backend/backend_config.h"
#include "GPTSoVITS/model/device.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS::Model {

/**
 * @brief Model Base Class
 */
class BaseModel {
public:
  virtual ~BaseModel() = default;

  /**
   * @brief Load model (legacy interface, uses default config)
   * @param model_path model path
   * @param device device
   * @param work_thread_num thread count
   */
  virtual bool Load(const std::string& model_path, const Device& device,
                    int work_thread_num) = 0;

  /**
   * @brief Load model with config (recommended interface)
   * @param model_path model path
   * @param config backend configuration
   */
  virtual bool Load(const std::string& model_path, const BackendConfig& config) = 0;

  /**
   * @brief Get current backend config
   */
  virtual const BackendConfig& GetConfig() const { return config_; }

  /**
   * @brief Inference
   * @param inputs inputs
   * @param outputs outputs
   */
  virtual void Forward(const std::unordered_map<std::string, Tensor*>& inputs,
                       std::unordered_map<std::string, std::unique_ptr<Tensor>>& outputs) = 0;

  virtual void Forward(const std::unordered_map<std::string, Tensor*>& inputs,
             std::vector<std::unique_ptr<Tensor>>& outputs) = 0;

  /**
   * @brief Inference with pre-allocated outputs (zero-copy support)
   *
   * This method accepts pre-allocated output tensors, avoiding memory allocation
   * during inference. This is critical for performance in generation loops.
   *
   * @param inputs inputs
   * @param outputs pre-allocated output tensors (will be written to)
   * @return true if successful
   */
  virtual bool ForwardWithPreallocatedOutput(
      const std::unordered_map<std::string, Tensor*>& inputs,
      std::unordered_map<std::string, Tensor*>& outputs) = 0;

  /**
   * @brief Get input names
   */
  virtual std::vector<std::string> GetInputNames() const = 0;

  /**
   * @brief Get output names
   */
  virtual std::vector<std::string> GetOutputNames() const = 0;

  /**
   * @brief Get input data type
   */
  virtual DataType GetInputDataType(const std::string& name) const = 0;

  /**
   * @brief Get output data type
   */
  virtual DataType GetOutputDataType(const std::string& name) const = 0;

  [[nodiscard]] Device GetDevice() const { return device_; }

protected:
  Device device_;
  BackendConfig config_;
};

}  // namespace GPTSoVITS::Model

#endif  // GPT_SOVITS_CPP_MODEL_BASE_H

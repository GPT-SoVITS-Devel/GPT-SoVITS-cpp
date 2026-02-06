//
// Created by 19254 on 2026/1/10.
//

#ifndef GPT_SOVITS_CPP_MODEL_BASE_H
#define GPT_SOVITS_CPP_MODEL_BASE_H

#include <string>
#include <unordered_map>
#include <vector>

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
   * @brief Load model
   * @param model_path model path
   * @param device device
   * @param work_thread_num thread count
   */
  virtual bool Load(const std::string& model_path, const Device& device,
                    int work_thread_num) = 0;

  /**
   * @brief Inference
   * @param inputs inputs
   * @param outputs outputs
   */
  virtual void Forward(const std::unordered_map<std::string, Tensor*>& inputs,
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
};

}  // namespace GPTSoVITS::Model

#endif  // GPT_SOVITS_CPP_MODEL_BASE_H

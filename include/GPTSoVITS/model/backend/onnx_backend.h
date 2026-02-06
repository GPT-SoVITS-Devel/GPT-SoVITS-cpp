#ifndef GPT_SOVITS_CPP_ONNX_BACKEND_H
#define GPT_SOVITS_CPP_ONNX_BACKEND_H

#include <memory>

#include "GPTSoVITS/model/base.h"

namespace Ort {
class Session;
class Env;
}  // namespace Ort

namespace GPTSoVITS::Model {

/**
 * @brief ONNX Runtime Backend
 */
class ONNXBackend : public BaseModel {
public:
  ONNXBackend();
  ~ONNXBackend() override;

  bool Load(const std::string& model_path, const Device& device,
            int work_thread_num) override;

  void Forward(const std::unordered_map<std::string, Tensor*>& inputs,
               std::unordered_map<std::string, Tensor*>& outputs) override;

  std::vector<std::string> GetInputNames() const override;
  std::vector<std::string> GetOutputNames() const override;
  DataType GetInputDataType(const std::string& name) const;
  DataType GetOutputDataType(const std::string& name) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace GPTSoVITS::Model

#endif  // GPT_SOVITS_CPP_ONNX_BACKEND_H

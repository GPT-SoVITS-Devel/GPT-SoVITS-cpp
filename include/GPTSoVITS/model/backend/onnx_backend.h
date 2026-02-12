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
               std::unordered_map<std::string, std::unique_ptr<Tensor>>&
                   outputs) override;
  void Forward(const std::unordered_map<std::string, Tensor*>& inputs,
               std::vector<std::unique_ptr<Tensor>>& outputs) override;
  std::vector<std::string> GetInputNames() const override;
  std::vector<std::string> GetOutputNames() const override;
  DataType GetInputDataType(const std::string& name) const;
  DataType GetOutputDataType(const std::string& name) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  /**
   * @brief 核心推理逻辑，独立于输出容器格式
   * * @param impl 后端实现指针
   * @param device 当前运行设备
   * @param inputs 输入张量映射
   * @param target_output_names 需要获取的输出名称列表
   * @return std::vector<std::unique_ptr<Tensor>> 按 target_output_names
   * 顺序排列的输出张量
   */
  std::vector<std::unique_ptr<Tensor>> InferCore(
      Impl* impl, const Device& device,
      const std::unordered_map<std::string, Tensor*>& inputs,
      const std::vector<std::string>& target_output_names);
};

}  // namespace GPTSoVITS::Model

#endif  // GPT_SOVITS_CPP_ONNX_BACKEND_H

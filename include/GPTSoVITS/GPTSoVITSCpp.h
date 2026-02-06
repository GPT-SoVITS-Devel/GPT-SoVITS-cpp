//
// Created by 19254 on 2026/1/17.
//

#ifndef GPT_SOVITS_CPP_GPTSOVITSCPP_H
#define GPT_SOVITS_CPP_GPTSOVITSCPP_H

#include <string>
#include <utility>

namespace GPTSoVITS {

class GPTSoVITS {
public:
  enum class BackendType : char {
    ONNX,
    TENSORRT,
  };
  /**
   * @param model_path 模型集合目录
   * @param backend_type 模型后端类型
   */
  explicit GPTSoVITS(std::string model_path,
                     BackendType backend_type = BackendType::ONNX)
      : m_backend_type(backend_type), m_model_path(std::move(model_path)) {};
  virtual ~GPTSoVITS() = default;

private:
  BackendType m_backend_type;
  std::string m_model_path;
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_GPTSOVITSCPP_H

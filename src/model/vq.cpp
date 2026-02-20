//
// Created by 19254 on 2026/2/10.
//
#include <GPTSoVITS/model/vq.h>

#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::Model {

std::unique_ptr<Tensor> VQModel::GetVQCodes(Tensor* ssl_content) {
  Tensor* sslPtr = nullptr;
  std::unique_ptr<Tensor> ssl;
  if (ssl_content->Type() == m_model->GetInputDataType("ssl_content")) {
    sslPtr = ssl_content;
  } else {
    ssl = ssl_content->ToType(m_model->GetInputDataType("ssl_content"));
    sslPtr = ssl.get();
  }

  std::unordered_map<std::string, Tensor*> inputs = {
      {"ssl_content", sslPtr},
  };
  std::unordered_map<std::string, std::unique_ptr<Tensor>> outputs;
  m_model->Forward(inputs, outputs);
  
  auto it = outputs.find("codes");
  if (it != outputs.end()) {
    return std::move(it->second);
  }
  
  // 返回第一个输出
  if (!outputs.empty()) {
    return std::move(outputs.begin()->second);
  }
  
  return nullptr;
};

}  // namespace GPTSoVITS::Model
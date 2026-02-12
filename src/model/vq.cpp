//
// Created by 19254 on 2026/2/10.
//
#include <GPTSoVITS/model/vq.h>

#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::Model {

std::unique_ptr<Tensor> VQModel::GetVQCodes(Tensor* ssl_content) {
  // Device model_device = ssl_content.GetDevice();
  // auto ssl_content =
  //     input_audio_16k->To(model_device, m_model->GetInputDataType("audio"));
  try {
    std::unordered_map<std::string, Tensor*> inputs = {
        {"ssl_content", ssl_content},
    };
    std::vector<std::unique_ptr<Tensor>> outputs;
    auto ips = m_model->GetInputNames();
    for (auto& i : ips) {
      PrintDebug("n: [{}]{}", (int)m_model->GetInputDataType(i), i);
    }
    m_model->Forward(inputs, outputs);
  } catch (const std::exception& e) {
    PrintError("e:{}", e.what());
  }
  return nullptr;
};

}  // namespace GPTSoVITS::Model
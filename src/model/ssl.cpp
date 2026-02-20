//
// Created by Huiyicc on 2026/2/8.
//
#include "GPTSoVITS/model/ssl.h"

namespace GPTSoVITS::Model {

std::unique_ptr<Tensor> SSLModel::GetSSLContent(
    const std::vector<float>& audio_16k) {
  std::vector<int64_t> shape = {1, static_cast<int64_t>(audio_16k.size())};
  auto input_audio_16k = Tensor::CreateFromHost(
      const_cast<float*>(audio_16k.data()), shape, DataType::kFloat32);
  Device model_device = m_model->GetDevice();
  auto wav16k_padded =
      input_audio_16k->To(model_device, m_model->GetInputDataType("audio"));
  std::unordered_map<std::string, Tensor*> inputs = {
      {"audio", wav16k_padded.get()},
  };
  std::unordered_map<std::string, std::unique_ptr<Tensor>> outputs;
  m_model->Forward(inputs, outputs);
  
  auto it = outputs.find("last_hidden_state");
  if (it != outputs.end()) {
    return std::move(it->second);
  }

  if (!outputs.empty()) {
    return std::move(outputs.begin()->second);
  }
  
  return nullptr;
}

}  // namespace GPTSoVITS::Model

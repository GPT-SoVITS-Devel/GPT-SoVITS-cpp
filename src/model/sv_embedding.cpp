//
// Created by 回忆 on 2026/2/19.
//
#include "GPTSoVITS/model/sv_embedding.h"

namespace GPTSoVITS::Model {

std::unique_ptr<Tensor> SVEmbeddingModel::ComputeEmbedding(
    const std::vector<float>& audio_16k) const {
  std::vector<int64_t> shape = {1, static_cast<int64_t>(audio_16k.size())};
  auto input_audio_16k = Tensor::CreateFromHost(
      const_cast<float*>(audio_16k.data()), shape, DataType::kFloat32);
  Device model_device = m_model->GetDevice();
  auto wav32k_padded =
      input_audio_16k->To(model_device, m_model->GetInputDataType("audio"));
  std::unordered_map<std::string, Tensor*> inputs = {
      {"audio", wav32k_padded.get()},
  };

  std::unordered_map<std::string, std::unique_ptr<Tensor>> outputs;
  m_model->Forward(inputs, outputs);
  return std::move(outputs["sv_embedding"]);
}

}  // namespace GPTSoVITS::Model

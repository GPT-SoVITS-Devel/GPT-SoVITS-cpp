//
// Created by 回忆 on 2026/2/19.
//
#include "GPTSoVITS/model/spectrogram.h"

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::Model {

std::unique_ptr<Tensor> SpectrogramModel::ComputeSpec(
    const std::vector<float>& audio_32k) {
  std::vector<int64_t> shape = {1, static_cast<int64_t>(audio_32k.size())};
  auto input_audio_32k = Tensor::CreateFromHost(
      std::vector<float>(audio_32k).data(), shape, DataType::kFloat32);
  Device model_device = m_model->GetDevice();
  auto wav32k_padded =
      input_audio_32k->To(model_device, m_model->GetInputDataType("audio"));
  std::unordered_map<std::string, Tensor*> inputs = {
      {"audio", wav32k_padded.get()},
  };

  std::unordered_map<std::string, std::unique_ptr<Tensor>> outputs;
  m_model->Forward(inputs, outputs);
  return std::move(outputs["spectrogram"]);
}

}  // namespace GPTSoVITS::Model

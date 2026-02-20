//
// Created by Huiyicc on 2026/2/17.
//

#include "GPTSoVITS/model/sovits.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::Model {

std::unique_ptr<Tensor> SoVITSModel::GenerateTensor(Tensor* pred_semantic,
                                                    Tensor* text_seq,
                                                    Tensor* refer_spec,
                                                    Tensor* sv_emb,
                                                    float noise_scale,
                                                    float speed) {

  // Ensure inputs are on the correct device and have correct types
  Device model_device = m_model->GetDevice();

  // Prepare pred_semantic (convert to int64 if needed)
  std::unique_ptr<Tensor> pred_semantic_converted;
  Tensor* pred_semantic_ptr = nullptr;
  if (pred_semantic->GetDeviceType() != model_device.type ||
      pred_semantic->Type() != m_model->GetInputDataType("pred_semantic")) {
    pred_semantic_converted = pred_semantic->To(model_device, m_model->GetInputDataType("pred_semantic"));
    pred_semantic_ptr = pred_semantic_converted.get();
  } else {
    pred_semantic_ptr = pred_semantic;
  }

  // Prepare text_seq
  std::unique_ptr<Tensor> text_seq_converted;
  Tensor* text_seq_ptr = nullptr;
  if (text_seq->GetDeviceType() != model_device.type ||
      text_seq->Type() != m_model->GetInputDataType("text_seq")) {
    text_seq_converted = text_seq->To(model_device, m_model->GetInputDataType("text_seq"));
    text_seq_ptr = text_seq_converted.get();
  } else {
    text_seq_ptr = text_seq;
  }

  // Prepare refer_spec
  std::unique_ptr<Tensor> refer_spec_converted;
  Tensor* refer_spec_ptr = nullptr;
  if (refer_spec->GetDeviceType() != model_device.type ||
      refer_spec->Type() != m_model->GetInputDataType("refer_spec")) {
    refer_spec_converted = refer_spec->To(model_device, m_model->GetInputDataType("refer_spec"));
    refer_spec_ptr = refer_spec_converted.get();
  } else {
    refer_spec_ptr = refer_spec;
  }

  // Prepare sv_emb
  std::unique_ptr<Tensor> sv_emb_converted;
  Tensor* sv_emb_ptr = nullptr;
  if (sv_emb->GetDeviceType() != model_device.type ||
      sv_emb->Type() != m_model->GetInputDataType("sv_emb")) {
    sv_emb_converted = sv_emb->To(model_device, m_model->GetInputDataType("sv_emb"));
    sv_emb_ptr = sv_emb_converted.get();
  } else {
    sv_emb_ptr = sv_emb;
  }

  // Prepare noise_scale and speed tensors
  auto noise_scale_tensor = Tensor::Empty({1}, DataType::kFloat32, model_device);
  auto speed_tensor = Tensor::Empty({1}, DataType::kFloat32, model_device);
  noise_scale_tensor->At<float>(0) = noise_scale;
  speed_tensor->At<float>(0) = speed;

  // Prepare inputs
  std::unordered_map<std::string, Tensor*> inputs = {
      {"pred_semantic", pred_semantic_ptr},
      {"text_seq", text_seq_ptr},
      {"refer_spec", refer_spec_ptr},
      {"sv_emb", sv_emb_ptr},
      {"noise_scale", noise_scale_tensor.get()},
      {"speed", speed_tensor.get()}
  };

  // Run inference
  std::unordered_map<std::string, std::unique_ptr<Tensor>> outputs;
  m_model->Forward(inputs, outputs);

  // Extract audio output
  if (outputs.find("audio") != outputs.end()) {
    return std::move(outputs["audio"]);
  } else {
    PrintError("SoVITS: missing 'audio' output");
    return nullptr;
  }
}

std::vector<float> SoVITSModel::Generate(Tensor* pred_semantic,
                                          Tensor* text_seq,
                                          Tensor* refer_spec,
                                          Tensor* sv_emb,
                                          float noise_scale,
                                          float speed) {

  auto audio_tensor = GenerateTensor(pred_semantic, text_seq, refer_spec, sv_emb, noise_scale, speed);

  if (!audio_tensor) {
    return {};
  }

  // Convert to CPU and extract data
  auto audio_cpu = audio_tensor->ToCPU();
  auto audio_ptr = audio_cpu->Data<float>();
  int64_t num_samples = audio_cpu->ElementCount();

  return std::vector<float>(audio_ptr, audio_ptr + num_samples);
}

}  // namespace GPTSoVITS::Model
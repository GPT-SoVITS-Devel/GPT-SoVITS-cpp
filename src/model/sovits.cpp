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

  // Prepare inputs
  std::unordered_map<std::string, Tensor*> inputs;
  inputs["pred_semantic"] = pred_semantic_ptr;
  inputs["text_seq"] = text_seq_ptr;
  inputs["refer_spec"] = refer_spec_ptr;

  auto input_names = m_model->GetInputNames();
  auto has_input = [&](const std::string& name) {
    return std::find(input_names.begin(), input_names.end(), name) !=
           input_names.end();
  };

  // Optional: sv_emb
  std::unique_ptr<Tensor> sv_emb_tensor;
  if (has_input("sv_emb") && sv_emb) {
    sv_emb_tensor = sv_emb->To(model_device, m_model->GetInputDataType("sv_emb"));
    inputs["sv_emb"] = sv_emb_tensor.get();
  }

  // Optional: noise_scale
  std::unique_ptr<Tensor> noise_scale_tensor;
  if (has_input("noise_scale")) {
    auto noise_scale_cpu = Tensor::Empty({1}, DataType::kFloat32, Device(DeviceType::kCPU));
    noise_scale_cpu->At<float>(0) = noise_scale;
    noise_scale_tensor = noise_scale_cpu->To(model_device, m_model->GetInputDataType("noise_scale"));
    inputs["noise_scale"] = noise_scale_tensor.get();
  }

  // Optional: speed
  std::unique_ptr<Tensor> speed_tensor;
  if (has_input("speed")) {
    auto speed_cpu = Tensor::Empty({1}, DataType::kFloat32, Device(DeviceType::kCPU));
    speed_cpu->At<float>(0) = speed;
    speed_tensor = speed_cpu->To(model_device, m_model->GetInputDataType("speed"));
    inputs["speed"] = speed_tensor.get();
  }

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

  // Convert to CPU and extract data (ensure float32)
  auto audio_cpu = audio_tensor->To(Device(DeviceType::kCPU), DataType::kFloat32);
  auto audio_ptr = audio_cpu->Data<float>();
  int64_t num_samples = audio_cpu->ElementCount();

  return std::vector<float>(audio_ptr, audio_ptr + num_samples);
}

}  // namespace GPTSoVITS::Model
//
// Created by Huiyicc on 2026/2/17.
//

#include "GPTSoVITS/model/gpt_encoder.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::Model {

GPTEncoderOutput GPTEncoderModel::Encode(
    Tensor* phoneme_ids,
    Tensor* phoneme_ids_len,
    Tensor* prompts,
    Tensor* bert_feature) {

  GPTEncoderOutput output;

  // Ensure inputs are on the correct device and have correct types
  Device model_device = m_model->GetDevice();

  // Prepare phoneme_ids
  std::unique_ptr<Tensor> phoneme_ids_converted;
  Tensor* phoneme_ids_ptr = nullptr;
  if (phoneme_ids->GetDeviceType() != model_device.type ||
      phoneme_ids->Type() != m_model->GetInputDataType("phoneme_ids")) {
    phoneme_ids_converted = phoneme_ids->To(model_device, m_model->GetInputDataType("phoneme_ids"));
    phoneme_ids_ptr = phoneme_ids_converted.get();
  } else {
    phoneme_ids_ptr = phoneme_ids;
  }

  // // Prepare phoneme_ids_len
  // std::unique_ptr<Tensor> phoneme_ids_len_converted;
  // Tensor* phoneme_ids_len_ptr = nullptr;
  // if (phoneme_ids_len->GetDeviceType() != model_device.type ||
  //     phoneme_ids_len->Type() != m_model->GetInputDataType("phoneme_ids_len")) {
  //   phoneme_ids_len_converted = phoneme_ids_len->To(model_device, m_model->GetInputDataType("phoneme_ids_len"));
  //   phoneme_ids_len_ptr = phoneme_ids_len_converted.get();
  // } else {
  //   phoneme_ids_len_ptr = phoneme_ids_len;
  // }

  // Prepare prompts (convert to int64 if needed)
  std::unique_ptr<Tensor> prompts_converted;
  Tensor* prompts_ptr = nullptr;
  if (prompts->GetDeviceType() != model_device.type ||
      prompts->Type() != m_model->GetInputDataType("prompts")) {
    prompts_converted = prompts->To(model_device, m_model->GetInputDataType("prompts"));
    prompts_ptr = prompts_converted.get();
  } else {
    prompts_ptr = prompts;
  }

  // Prepare bert_feature
  std::unique_ptr<Tensor> bert_feature_converted;
  Tensor* bert_feature_ptr = nullptr;
  if (bert_feature->GetDeviceType() != model_device.type ||
      bert_feature->Type() != m_model->GetInputDataType("bert_feature")) {
    bert_feature_converted = bert_feature->To(model_device, m_model->GetInputDataType("bert_feature"));
    bert_feature_ptr = bert_feature_converted.get();
  } else {
    bert_feature_ptr = bert_feature;
  }

  // Prepare inputs
  std::unordered_map<std::string, Tensor*> inputs = {
      {"phoneme_ids", phoneme_ids_ptr},
      // {"phoneme_ids_len", phoneme_ids_len_ptr},
      {"prompts", prompts_ptr},
      {"bert_feature", bert_feature_ptr}
  };

  // Run inference
  std::unordered_map<std::string, std::unique_ptr<Tensor>> outputs;
  m_model->Forward(inputs, outputs);

  // Extract outputs
  if (outputs.find("topk_values") != outputs.end()) {
    output.topk_values = std::move(outputs["topk_values"]);
  } else {
    PrintError("GPT Encoder: missing 'topk_values' output");
  }

  if (outputs.find("topk_indices") != outputs.end()) {
    output.topk_indices = std::move(outputs["topk_indices"]);
  } else {
    PrintError("GPT Encoder: missing 'topk_indices' output");
  }

  if (outputs.find("k_cache") != outputs.end()) {
    output.k_cache = std::move(outputs["k_cache"]);
    // Update cache metadata
    auto cache_shape = output.k_cache->Shape();
    if (cache_shape.size() >= 5) {
      m_num_layers = cache_shape[0];
      m_num_heads = cache_shape[2];
      m_head_dim = cache_shape[4];
      PrintInfo("GPT Encoder cache detected: num_layers={}, num_heads={}, head_dim={}",
                m_num_layers, m_num_heads, m_head_dim);
    }
  } else {
    PrintError("GPT Encoder: missing 'k_cache' output");
  }

  if (outputs.find("v_cache") != outputs.end()) {
    output.v_cache = std::move(outputs["v_cache"]);
  } else {
    PrintError("GPT Encoder: missing 'v_cache' output");
  }

  if (outputs.find("x_len") != outputs.end()) {
    output.x_len = std::move(outputs["x_len"]);
  } else {
    PrintError("GPT Encoder: missing 'x_len' output");
  }

  if (outputs.find("y_len") != outputs.end()) {
    output.y_len = std::move(outputs["y_len"]);
  } else {
    PrintError("GPT Encoder: missing 'y_len' output");
  }

  return output;
}

}  // namespace GPTSoVITS::Model
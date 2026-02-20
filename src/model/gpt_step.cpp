//
// Created by Huiyicc on 2026/2/17.
//

#include "GPTSoVITS/model/gpt_step.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::Model {

GPTStepOutput GPTStepModel::Step(Tensor* samples,
                                 Tensor* k_cache,
                                 Tensor* v_cache,
                                 Tensor* idx,
                                 Tensor* x_len,
                                 Tensor* y_len) {

  GPTStepOutput output;

  // Ensure inputs are on the correct device and have correct types
  Device model_device = m_model->GetDevice();

  // Prepare samples
  std::unique_ptr<Tensor> samples_converted;
  Tensor* samples_ptr = nullptr;
  if (samples->GetDeviceType() != model_device.type ||
      samples->Type() != m_model->GetInputDataType("samples")) {
    samples_converted = samples->To(model_device, m_model->GetInputDataType("samples"));
    samples_ptr = samples_converted.get();
  } else {
    samples_ptr = samples;
  }

  // Prepare k_cache
  std::unique_ptr<Tensor> k_cache_converted;
  Tensor* k_cache_ptr = nullptr;
  if (k_cache->GetDeviceType() != model_device.type ||
      k_cache->Type() != m_model->GetInputDataType("k_cache")) {
    k_cache_converted = k_cache->To(model_device, m_model->GetInputDataType("k_cache"));
    k_cache_ptr = k_cache_converted.get();
  } else {
    k_cache_ptr = k_cache;
  }

  // Prepare v_cache
  std::unique_ptr<Tensor> v_cache_converted;
  Tensor* v_cache_ptr = nullptr;
  if (v_cache->GetDeviceType() != model_device.type ||
      v_cache->Type() != m_model->GetInputDataType("v_cache")) {
    v_cache_converted = v_cache->To(model_device, m_model->GetInputDataType("v_cache"));
    v_cache_ptr = v_cache_converted.get();
  } else {
    v_cache_ptr = v_cache;
  }

  // Prepare idx
  std::unique_ptr<Tensor> idx_converted;
  Tensor* idx_ptr = nullptr;
  if (idx->GetDeviceType() != model_device.type ||
      idx->Type() != m_model->GetInputDataType("idx")) {
    idx_converted = idx->To(model_device, m_model->GetInputDataType("idx"));
    idx_ptr = idx_converted.get();
  } else {
    idx_ptr = idx;
  }

  // Prepare x_len
  std::unique_ptr<Tensor> x_len_converted;
  Tensor* x_len_ptr = nullptr;
  if (x_len->GetDeviceType() != model_device.type ||
      x_len->Type() != m_model->GetInputDataType("x_len")) {
    x_len_converted = x_len->To(model_device, m_model->GetInputDataType("x_len"));
    x_len_ptr = x_len_converted.get();
  } else {
    x_len_ptr = x_len;
  }

  // Prepare y_len
  std::unique_ptr<Tensor> y_len_converted;
  Tensor* y_len_ptr = nullptr;
  if (y_len->GetDeviceType() != model_device.type ||
      y_len->Type() != m_model->GetInputDataType("y_len")) {
    y_len_converted = y_len->To(model_device, m_model->GetInputDataType("y_len"));
    y_len_ptr = y_len_converted.get();
  } else {
    y_len_ptr = y_len;
  }

  // Prepare inputs
  std::unordered_map<std::string, Tensor*> inputs = {
      {"samples", samples_ptr},
      {"k_cache", k_cache_ptr},
      {"v_cache", v_cache_ptr},
      {"idx", idx_ptr},
      {"x_len", x_len_ptr},
      {"y_len", y_len_ptr}
  };

  // Run inference
  std::unordered_map<std::string, std::unique_ptr<Tensor>> outputs;
  m_model->Forward(inputs, outputs);

  // Extract outputs
  if (outputs.find("topk_values") != outputs.end()) {
    output.topk_values = std::move(outputs["topk_values"]);
  } else {
    PrintError("GPT Step: missing 'topk_values' output");
  }

  if (outputs.find("topk_indices") != outputs.end()) {
    output.topk_indices = std::move(outputs["topk_indices"]);
  } else {
    PrintError("GPT Step: missing 'topk_indices' output");
  }

  if (outputs.find("k_cache_new") != outputs.end()) {
    output.k_cache_new = std::move(outputs["k_cache_new"]);
  } else {
    PrintError("GPT Step: missing 'k_cache_new' output");
  }

  if (outputs.find("v_cache_new") != outputs.end()) {
    output.v_cache_new = std::move(outputs["v_cache_new"]);
  } else {
    PrintError("GPT Step: missing 'v_cache_new' output");
  }

  return output;
}

bool GPTStepModel::StepWithIOBinding(Tensor* samples,
                                     Tensor* k_cache_in,
                                     Tensor* v_cache_in,
                                     Tensor* k_cache_out,
                                     Tensor* v_cache_out,
                                     Tensor* idx,
                                     Tensor* x_len,
                                     Tensor* y_len,
                                     Tensor* topk_values_out,
                                     Tensor* topk_indices_out) {

  // This is a stub implementation for IO binding
  // For actual zero-copy, we need to extend the ONNXBackend to support
  // IO binding similar to Python's io_binding.bind_ortvalue_input/output
  //
  // The implementation would involve:
  // 1. Pre-allocating all tensors on the device
  // 2. Using OrtIoBinding from ONNX Runtime to bind inputs/outputs directly
  // 3. Calling RunWithIOBinding instead of regular Run
  //
  // For now, we'll use the regular Step method which involves copies
  // TODO: Implement proper IO binding in ONNXBackend

  GPTStepOutput output = Step(samples, k_cache_in, v_cache_in, idx, x_len, y_len);

  // Copy outputs to pre-allocated buffers
  if (topk_values_out && output.topk_values) {
    auto& out_values = *topk_values_out;
    auto& src_values = *output.topk_values;
    std::memcpy(out_values.Data<void>(), src_values.Data<void>(), src_values.ByteSize());
  }

  if (topk_indices_out && output.topk_indices) {
    auto& out_indices = *topk_indices_out;
    auto& src_indices = *output.topk_indices;
    std::memcpy(out_indices.Data<void>(), src_indices.Data<void>(), src_indices.ByteSize());
  }

  if (k_cache_out && output.k_cache_new) {
    auto& out_cache = *k_cache_out;
    auto& src_cache = *output.k_cache_new;
    std::memcpy(out_cache.Data<void>(), src_cache.Data<void>(), src_cache.ByteSize());
  }

  if (v_cache_out && output.v_cache_new) {
    auto& out_cache = *v_cache_out;
    auto& src_cache = *output.v_cache_new;
    std::memcpy(out_cache.Data<void>(), src_cache.Data<void>(), src_cache.ByteSize());
  }

  return true;
}

}  // namespace GPTSoVITS::Model
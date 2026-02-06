//
// Created by 19254 on 2026/2/5.
//
#include <GPTSoVITS/model/CNBertModel.h>

#include <fstream>

#include "GPTSoVITS/G2P/Base.h"
#include "GPTSoVITS/Utils/exception.h"

namespace GPTSoVITS::Model {

CNBertModel::CNBertModel() = default;

BertModel::EncodeResult CNBertModel::EncodeText(const std::string& text) {
  auto rust_res = m_tokenzer->EncodeEx(text, false);
  EncodeResult res;
  res.TokenIds.reserve(rust_res.TokenIds.size());
  res.TokenTypeIds.reserve(rust_res.TokenTypeIds.size());
  res.Masks.reserve(rust_res.Masks.size());
  for (const auto id : rust_res.TokenIds)
    res.TokenIds.push_back(static_cast<int64_t>(id));
  for (const auto id : rust_res.TokenTypeIds)
    res.TokenTypeIds.push_back(static_cast<int64_t>(id));
  for (const auto id : rust_res.Masks)
    res.Masks.push_back(static_cast<int64_t>(id));
  return res;
}

std::unique_ptr<Tensor> CNBertModel::GetBertFeature(
    const std::string& text, const G2P::G2PRes& g2p_info) {
  auto encode_res = EncodeText(text);

  std::vector<int64_t> shape = {
      1, static_cast<int64_t>(encode_res.TokenIds.size())};

  auto input_ids_base = Tensor::CreateFromHost(
      encode_res.TokenIds.data(), shape, DataType::kInt64);
  auto attention_mask_base = Tensor::CreateFromHost(
      encode_res.Masks.data(), shape, DataType::kInt64);
  auto token_type_ids_base = Tensor::CreateFromHost(
      encode_res.TokenTypeIds.data(), shape, DataType::kInt64);

  Device model_device = m_model->GetDevice();
  auto in_ids = input_ids_base->To(model_device, m_model->GetInputDataType("input_ids"));
  auto in_mask = attention_mask_base->To(model_device, m_model->GetInputDataType("attention_mask"));
  auto in_type = token_type_ids_base->To(model_device, m_model->GetInputDataType("token_type_ids"));

  std::unordered_map<std::string, Tensor*> inputs = {
      {"input_ids", in_ids.get()},
      {"attention_mask", in_mask.get()},
      {"token_type_ids", in_type.get()}};

  std::unordered_map<std::string, Tensor*> outputs;
  m_model->Forward(inputs, outputs);

  auto it = outputs.find("hidden_states");
  if (it == outputs.end()) {
    for (auto& pair : outputs) delete pair.second;
    THROW_ERRORN("BERT model output 'hidden_states' not found.");
  }

  std::unique_ptr<Tensor> raw_hidden(it->second);
  for (auto& pair : outputs) {
    if (pair.first != "hidden_states") delete pair.second;
  }

  // 统一转为 Float32 处理 (CPU)
  auto cpu_hidden = raw_hidden->To(Device(DeviceType::kCPU), DataType::kFloat32);
  float* src_ptr = cpu_hidden->Data<float>();
  int64_t L = cpu_hidden->Shape()[1];
  int64_t D = cpu_hidden->Shape()[2];  // 1024

  int64_t total_phones = 0;
  for (int count : g2p_info.word2ph) total_phones += count;

  std::vector<int64_t> out_shape = {D, total_phones};
  auto out_tensor =
      Tensor::Empty(out_shape, DataType::kFloat32, Device(DeviceType::kCPU));
  float* dst_ptr = out_tensor->Data<float>();

  // res = hidden_states[0][1:-1] -> shape (L, 1024)
  if (g2p_info.word2ph.size() != static_cast<size_t>(L)) {
    THROW_ERRORN(
        "word2ph size ({}) does not match the shape of hidden_states (L={}) "
        "for text: '{}'. ",
        g2p_info.word2ph.size(), L, text);
  }

  int64_t phone_offset_base = 0;
  std::vector<int64_t> phone_offsets;
  phone_offsets.reserve(g2p_info.word2ph.size());
  for (int count : g2p_info.word2ph) {
    phone_offsets.push_back(phone_offset_base);
    phone_offset_base += count;
  }

  for (int64_t d = 0; d < D; ++d) {
    float* dst_row = dst_ptr + d * total_phones;
    for (size_t i = 0; i < g2p_info.word2ph.size(); ++i) {
      if (i + 1 >= static_cast<size_t>(L - 1)) break;
      float val = src_ptr[(i + 1) * D + d];
      int repeat_count = g2p_info.word2ph[i];
      int64_t offset = phone_offsets[i];
      for (int r = 0; r < repeat_count; ++r) {
        dst_row[offset + r] = val;
      }
    }
  }

  return out_tensor;
}

}  // namespace GPTSoVITS::Model
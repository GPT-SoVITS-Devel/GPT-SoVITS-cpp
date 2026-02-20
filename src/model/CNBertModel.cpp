//
// Created by Huiyicc on 2026/2/5.
//
#include <GPTSoVITS/model/CNBertModel.h>

#include <fstream>

#include "GPTSoVITS/G2P/Base.h"
#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/plog.h"

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

  std::unordered_map<std::string, std::unique_ptr<Tensor>> outputs;
  m_model->Forward(inputs, outputs);

  auto it = outputs.find("hidden_states");
  if (it == outputs.end()) {
    THROW_ERRORN("BERT model output 'hidden_states' not found.");
  }

  std::unique_ptr<Tensor> raw_hidden = std::move(it->second);

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
  // 初始化为0，防止word2ph与BERT token数量不一致导致未初始化垃圾值(如NaN)
  std::memset(dst_ptr, 0, out_tensor->ByteSize());

  // res = hidden_states[0][1:-1] -> shape (L-2, 1024), skipping [CLS] and [SEP]
  if (L < 2) {
    THROW_ERRORN("BERT sequence too short (L={}), need at least [CLS] + [SEP].", L);
  }
  if (static_cast<int64_t>(g2p_info.word2ph.size()) != L - 2) {
    PrintWarn("word2ph size ({}) differs from expected L-2 ({}), text: '{}'. This may cause incorrect BERT features.",
              g2p_info.word2ph.size(), L - 2, text);
    // 不再抛出异常，而是继续处理（可能是tokenizer差异）
  }

  int64_t phone_offset_base = 0;
  std::vector<int64_t> phone_offsets;
  phone_offsets.reserve(g2p_info.word2ph.size());
  for (int count : g2p_info.word2ph) {
    phone_offsets.push_back(phone_offset_base);
    phone_offset_base += count;
  }

  // res = hidden_states[0][1:-1] -> skip [CLS] (index 0) and [SEP] (last index)
  // word2ph should have size = L - 2 (excluding both [CLS] and [SEP])
  // This matches Python: res = hidden_states[0][1:-1]
  int64_t valid_hidden_len = L - 2;  // Exclude [CLS] and [SEP]
  int64_t max_word_index = std::min(static_cast<int64_t>(g2p_info.word2ph.size()), valid_hidden_len);

  // 检查word2ph大小是否正确
  if (static_cast<int64_t>(g2p_info.word2ph.size()) > valid_hidden_len) {
    PrintWarn("word2ph size ({}) > valid_hidden_len ({}), truncating to {} elements. This may indicate tokenizer mismatch.",
              g2p_info.word2ph.size(), valid_hidden_len, valid_hidden_len);
  }

  for (int64_t d = 0; d < D; ++d) {
    float* dst_row = dst_ptr + d * total_phones;
    for (int64_t i = 0; i < max_word_index; ++i) {
      // 跳过[CLS] token at index 0
      // Python使用 [1:-1]，所以对应hidden_states[0][i+1]，其中i从0到valid_hidden_len-1
      // 即访问hidden_states[0][1]到hidden_states[0][L-2]（跳过最后的[SEP]）
      int64_t hidden_idx = i + 1;
      // 确保不越界: hidden_idx < L - 1 (skip [SEP])
      if (hidden_idx >= L - 1) {
        // 安全检查，理论上不应该到达这里
        PrintError("Hidden index {} out of valid range (L-1={}), skipping", hidden_idx, L - 1);
        continue;
      }
      float val = src_ptr[hidden_idx * D + d];
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
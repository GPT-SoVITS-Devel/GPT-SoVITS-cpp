//
// Created by Huiyicc on 2026/2/5.
//
#include <GPTSoVITS/model/CNBertModel.h>

#include <fstream>

#include "GPTSoVITS/G2P/Base.h"
#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/plog.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xview.hpp"

namespace GPTSoVITS::Model {

CNBertModel::CNBertModel() = default;

BertModel::EncodeResult CNBertModel::EncodeText(const std::string& text) {
  // no [CLS] and [SEP]
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

  Tensor* hidden_tensor = it->second.get();
  auto out_shape = hidden_tensor->Shape();

  if (out_shape.size() != 3 || out_shape[0] != 1) {
    THROW_ERRORN("Expected hidden_states shape [1, seq_len, hidden_size]");
  }

  int64_t seq_len = out_shape[1];
  int64_t hidden_size = out_shape[2];
  const auto& word2ph = g2p_info.word2ph;

  if (seq_len != word2ph.size()) {
    THROW_ERRORN("Dimension mismatch: seq_len {} vs word2ph {}", seq_len, word2ph.size());
  }

  std::unique_ptr<Tensor> cpu_float_owner;
  const float* src_ptr = nullptr;

  if (hidden_tensor->IsCPU() && hidden_tensor->Type() == DataType::kFloat32) {
    src_ptr = hidden_tensor->Data<float>();
  } else {
    cpu_float_owner = hidden_tensor->To(Device(DeviceType::kCPU), DataType::kFloat32);
    src_ptr = cpu_float_owner->Data<float>();
  }

  int64_t total_phones = 0;
  for (int count : word2ph) {
    total_phones += count;
  }

  auto final_tensor = Tensor::Empty(
      {hidden_size, total_phones}, DataType::kFloat32, Device(DeviceType::kCPU));
  float* dst_ptr = final_tensor->Data<float>();

  // 寻址填充
  // src: [seq_len, hidden_size]
  // dst: [hidden_size, total_phones]
  for (int64_t h = 0; h < hidden_size; ++h) {
    float* dst_row = dst_ptr + h * total_phones;
    int64_t phone_idx = 0;

    for (int64_t s = 0; s < seq_len; ++s) {
      float val = src_ptr[s * hidden_size + h];

      int repeat = word2ph[s];
      if (repeat > 0) {
        std::fill_n(dst_row + phone_idx, repeat, val);
        phone_idx += repeat;
      }
    }
  }

  return final_tensor->ToDevice(m_model->GetDevice());
}

}  // namespace GPTSoVITS::Model
//
// Created by 19254 on 2026/2/5.
//
#include <GPTSoVITS/model/CNBertModel.h>

#include <fstream>

#include "GPTSoVITS/Utils/exception.h"

namespace GPTSoVITS::Model {

CNBertModel::CNBertModel() = default;

BertModel::EncodeResult CNBertModel::EncodeText(const std::string& text) {
  auto rust_res = m_tokenzer->EncodeEx(text, true);
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
    const std::string& text, const std::vector<int>& word2ph) {
  auto encode_res = EncodeText(text);

  std::vector<int64_t> shape = {
      1, static_cast<int64_t>(encode_res.TokenIds.size())};
  auto input_ids =
      Tensor::Empty(shape, DataType::kInt64, Device(DeviceType::kCPU));
  std::memcpy(input_ids->Data(), encode_res.TokenIds.data(),
              input_ids->ByteSize());

  auto attention_mask =
      Tensor::Empty(shape, DataType::kInt64, Device(DeviceType::kCPU));
  std::memcpy(attention_mask->Data(), encode_res.Masks.data(),
              attention_mask->ByteSize());

  auto token_type_ids =
      Tensor::Empty(shape, DataType::kInt64, Device(DeviceType::kCPU));
  std::memcpy(token_type_ids->Data(), encode_res.TokenTypeIds.data(),
              token_type_ids->ByteSize());

  // 移动到模型所在设备
  Device model_device = m_model->GetDevice();
  auto in_ids = input_ids->ToDevice(model_device);
  auto in_mask = attention_mask->ToDevice(model_device);
  auto in_type = token_type_ids->ToDevice(model_device);

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
  // 清理 outputs 中其他可能存在的 Tensor
  for (auto& pair : outputs) {
    if (pair.first != "hidden_states") delete pair.second;
  }
  // raw_hidden is (1, L, 1024)
  // 我们需要去掉 [CLS] 和 [SEP], 即 [0][1:-1]
  // 然后根据 word2ph 重复

  auto cpu_hidden = raw_hidden->ToCPU();
  float* src_ptr = cpu_hidden->Data<float>();
  int64_t L = cpu_hidden->Shape()[1];
  int64_t D = cpu_hidden->Shape()[2];  // 1024

  int64_t total_phones = 0;
  for (int count : word2ph) total_phones += count;

  std::vector<int64_t> out_shape = {D, total_phones};
  auto out_tensor =
      Tensor::Empty(out_shape, DataType::kFloat32, Device(DeviceType::kCPU));
  float* dst_ptr = out_tensor->Data<float>();


  // res = hidden_states[0][1:-1] -> shape (L-2, 1024)
  // word2ph size 应该等于 L-2
  if (word2ph.size() != static_cast<size_t>(L - 2)) {
    THROW_ERRORN("word2ph does not match the shape of hidden_states");
  }

  // 填充逻辑以支持转置存贮 (1024, seq_len)
  int64_t current_phone_idx = 0;
  for (size_t i = 0; i < word2ph.size(); ++i) {
    if (i + 1 >= static_cast<size_t>(L - 1)) break;
    float* current_word_feat = src_ptr + (i + 1) * D;
    int repeat_count = word2ph[i];
    for (int r = 0; r < repeat_count; ++r) {
      for (int64_t d = 0; d < D; ++d) {
        dst_ptr[d * total_phones + current_phone_idx] = current_word_feat[d];
      }
      current_phone_idx++;
    }
  }

  return out_tensor;
}

}  // namespace GPTSoVITS::Model
//
// Created by 19254 on 2026/2/5.
//
#include "GPTSoVITS/G2P/Pipline.h"

#include <boost/algorithm/string.hpp>

#include "GPTSoVITS/Text/LangDetect.h"
#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/model/device.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::G2P {

G2PPipline::~G2PPipline() = default;

void G2PPipline::RegisterLangProcess(
    const std::string& lang, std::unique_ptr<IG2P> g2p_process,
    std::unique_ptr<Model::BertModel> bert_model, bool warm_up) {
  m_lang_process[lang] = std::move(g2p_process);
  if (bert_model) {
    m_bert_models[lang] = std::move(bert_model);
  }
  if (warm_up) {
    PrintInfo("warm up lang:{}", lang);
    m_lang_process[lang]->WarmUp();
  }
}

void G2PPipline::SetDefaultLang(const std::string& default_lang) {
  m_default_lang = default_lang;
};

std::string G2PPipline::GetLang(const std::string& lang) {
  if (m_lang_process.empty()) {
    THROW_ERRORN("g2p lang process empty");
  }

  auto iter = m_lang_process.find(lang);
  if (iter != m_lang_process.end()) {
    return iter->first;
  }

  // 回退默认
  iter = m_lang_process.find(m_default_lang);
  if (iter != m_lang_process.end()) {
    return iter->first;
  }

  return m_lang_process.begin()->first;
}

const IG2P* G2PPipline::GetG2P(const std::string& lang,
                               const std::string& default_lang) {
  if (m_lang_process.empty()) {
    THROW_ERRORN("g2p lang process empty");
  }
  auto iter = m_lang_process.find(lang);
  if (iter != m_lang_process.end()) {
    return iter->second.get();
  }
  std::string_view dLang = default_lang.empty() ? m_default_lang : default_lang;
  iter = m_lang_process.find(std::string(dLang));
  if (iter != m_lang_process.end()) {
    return iter->second.get();
  }
  // 未找到
  // 使用默认
  iter = m_lang_process.find(m_default_lang);
  if (iter != m_lang_process.end()) {
    return iter->second.get();
  }
  // 未找到
  // 使用第一个
  return m_lang_process.begin()->second.get();
};

std::shared_ptr<Bert::BertRes> G2PPipline::GetPhoneAndBert(
    const std::string& text, const std::string& default_lang) {
  auto htext = boost::trim_copy(text);
  auto [isReliable, de_lang] = Text::LangDetect::getInstance()->Detect(htext);
  if (!isReliable) {
    de_lang = default_lang;
  }
  auto detects =
      Text::LangDetect::getInstance()->DetectSplit(GetLang(de_lang), htext);

  std::vector<std::unique_ptr<Model::Tensor>> phone_tensors;
  std::vector<std::unique_ptr<Model::Tensor>> bert_tensors;

  for (auto& detectText : detects) {
    auto g2p = GetG2P(detectText.language);
    auto g2pRes = g2p->CleanText(detectText.sentence);

    // Phone ID Tensor
    std::vector<int64_t> phone_shape = {
        static_cast<int64_t>(g2pRes.phone_ids.size())};
    auto phone_tensor = Model::Tensor::Empty(
        phone_shape, Model::DataType::kInt64,
        ::GPTSoVITS::Model::Device(::GPTSoVITS::Model::DeviceType::kCPU));
    int64_t* phone_data = phone_tensor->Data<int64_t>();
    for (size_t i = 0; i < g2pRes.phone_ids.size(); ++i) {
      phone_data[i] = g2pRes.phone_ids[i];
    }
    phone_tensors.push_back(std::move(phone_tensor));

    // Bert Feature Tensor
    auto bert_iter = m_bert_models.find(detectText.language);
    if (bert_iter != m_bert_models.end()) {
      auto bert_feat =
          bert_iter->second->GetBertFeature(g2pRes.norm_text, g2pRes);
      bert_tensors.push_back(std::move(bert_feat));
    } else {
      // 如果没有BERT模型, 填充全零 (1024, seq_len)
      std::vector<int64_t> bert_shape = {
          1024, static_cast<int64_t>(g2pRes.phone_ids.size())};
      auto zero_bert = Model::Tensor::Empty(
          bert_shape, Model::DataType::kFloat32,
          ::GPTSoVITS::Model::Device(::GPTSoVITS::Model::DeviceType::kCPU));
      std::memset(zero_bert->Data(), 0, zero_bert->ByteSize());
      bert_tensors.push_back(std::move(zero_bert));
    }
  }

  // 拼接结果
  std::vector<Model::Tensor*> phone_ptrs;
  for (auto& t : phone_tensors) phone_ptrs.push_back(t.get());
  auto merged_phones = Model::Tensor::Concat(phone_ptrs, 0);

  std::vector<Model::Tensor*> bert_ptrs;
  for (auto& t : bert_tensors) bert_ptrs.push_back(t.get());
  auto merged_bert = Model::Tensor::Concat(bert_ptrs, 1);

  auto res = std::make_shared<Bert::BertRes>();
  res->PhoneSeq = std::move(merged_phones);
  res->BertSeq = std::move(merged_bert);
  return res;
}

}  // namespace GPTSoVITS::G2P

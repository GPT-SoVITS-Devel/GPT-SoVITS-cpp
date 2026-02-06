//
// Created by 19254 on 2026/2/5.
//
#include "GPTSoVITS/G2P/Pipline.h"

#include <boost/algorithm/string.hpp>

#include "GPTSoVITS/Text/LangDetect.h"
#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::G2P {

G2PPipline::~G2PPipline() = default;

void G2PPipline::RegisterLangProcess(
    const std::string& lang, std::unique_ptr<IG2P> g2p_process,
    std::unique_ptr<Model::BertModel> bert_model, bool warm_up) {
  m_lang_process[lang] = std::move(g2p_process);
  if (warm_up) {
    m_lang_process[lang]->WarmUp();
  }
}

void G2PPipline::SetDefaultLang(const std::string& default_lang) {
  m_default_lang = default_lang;
};

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
  iter = m_lang_process.find(dLang.data());
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
    const std::string& text, const std::string& default_lan) {
  auto htext = boost::trim_copy(text);
  auto [isReliable, de_lang] = Text::LangDetect::getInstance()->Detect(htext);
  if (!isReliable) {
    de_lang = default_lan;
  }
  auto detects = Text::LangDetect::getInstance()->DetectSplit(de_lang, htext);
  // std::vector<at::Tensor> PhoneSeqs;
  // std::vector<at::Tensor> BertSeqs;
  // for (auto& detectText : detects) {
  //   auto g2p = GetG2P(detectText.language);
  //   auto g2pRes = g2p->CleanText(detectText.sentence);
  //   auto bert = Bert::MakeFromLang(detectText.language);
  //   if (!bert) {
  //     PrintError("No Bert Model for {}\nSentence: {}", detectText.language,
  //                detectText.sentence);
  //   }
  //   auto encodeResult = (*bert)->Encode(g2pRes);
  //   PhoneSeqs.emplace_back(std::move(*encodeResult.PhoneSeq));
  //   BertSeqs.emplace_back(std::move(*encodeResult.BertSeq));
  // }
  // return std::make_shared<Bert::BertRes>(
  //     Bert::BertRes{std::make_shared<torch::Tensor>(
  //                       torch::cat({PhoneSeqs}, 1).to(*gpt.Device())),
  //                   std::make_shared<at::Tensor>(
  //                       torch::cat({BertSeqs}, 0).to(*gpt.Device()))});
}

}  // namespace GPTSoVITS::G2P

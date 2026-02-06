//
// Created by 19254 on 2026/2/5.
//

#ifndef GSV_CPP_PIPLINE_H
#define GSV_CPP_PIPLINE_H

#include <map>
#include <memory>
#include <string>

#include "GPTSoVITS/G2P/Base.h"
#include "GPTSoVITS/model/bert.h"

namespace GPTSoVITS::G2P {

class G2PPipline {
  std::map<std::string, std::unique_ptr<IG2P>> m_lang_process;
  std::map<std::string, std::unique_ptr<Model::BertModel>> m_bert_models;
  std::string m_default_lang;

public:
  virtual ~G2PPipline();

  void SetDefaultLang(const std::string& default_lang);

  const IG2P* GetG2P(const std::string& lang, const std::string& default_lang = "");

  /**
   * Register language process
   * @param lang language code (zh/en/ja)
   * @param g2p_process g2p processor
   * @param bert_model bert model instance
   * @param warm_up whether to warm up
   */
  void RegisterLangProcess(const std::string& lang,
                           std::unique_ptr<IG2P> g2p_process,
                           std::unique_ptr<Model::BertModel> bert_model,
                           bool warm_up = false);
  std::shared_ptr<Bert::BertRes> GetPhoneAndBert(
      const std::string& text, const std::string& default_lang = "");
};

}  // namespace GPTSoVITS::G2P

#endif  // GSV_CPP_PIPLINE_H

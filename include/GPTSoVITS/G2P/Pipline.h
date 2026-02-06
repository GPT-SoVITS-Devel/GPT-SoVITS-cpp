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
  std::string m_default_lang;

public:
  virtual ~G2PPipline();

  void SetDefaultLang(const std::string& default_lang);

  const IG2P* GetG2P(const std::string& lang, const std::string& default_lang = "");

  /**
   * 对指定语言注册对应的处理类
   * @param lang 用以注册支持的语言,语言代码一般为 ISO 639-1 风格(zh/en/ja)
   * @param g2p_process 用以g2p的处理类
   * @param bert_model 对应的 bert 实例
   * @param warm_up 是否预热
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

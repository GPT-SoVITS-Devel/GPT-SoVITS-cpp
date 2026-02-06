//
// Created by 19254 on 24-12-1.
//

#ifndef GPT_SOVITS_CPP_G2P_H
#define GPT_SOVITS_CPP_G2P_H

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace GPTSoVITS {
class GPTSoVITS;
}
namespace GPTSoVITS::Bert{
struct BertRes;
}
namespace GPTSoVITS::G2P {

struct G2PRes {
  std::vector<std::string> phones;
  std::vector<int> phone_ids;
  std::vector<int> word2ph;
  std::string norm_text;
};

class IG2P {

public:
  virtual ~IG2P() = default;
  virtual G2PRes CleanText(const std::string &text) const final;
  virtual void WarmUp() {};
private:
  // CleanText重写这个逻辑
  [[nodiscard]] virtual G2PRes _cleanText(const std::string &text) const = 0;
};

// std::shared_ptr<IG2P> MakeFromLang(const std::string&lang);
//
// std::shared_ptr<Bert::BertRes> GetPhoneAndBert(GPTSoVITS &gpt, const std::string &text,const std::string& lang="");
//


}

#endif //GPT_SOVITS_CPP_G2P_H

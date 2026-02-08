//
// Created by 19254 on 24-12-1.
//

#ifndef GPT_SOVITS_CPP_G2P_EN_H
#define GPT_SOVITS_CPP_G2P_EN_H

#include <GPTSoVITS/G2P/Base.h>
#include <string>
#include <map>

namespace GPTSoVITS::G2P {

namespace g2p_en {
std::vector<std::string> predict(const std::string &text);
}

class G2PEN : public IG2P {
public:
  void WarmUp() override;
  G2PRes _cleanText(const std::string &text) const override ;
};



};

#endif //GPT_SOVITS_CPP_G2P_EN_H

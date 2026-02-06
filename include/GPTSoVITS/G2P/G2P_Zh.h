//
// Created by 19254 on 2026/1/17.
//

#ifndef GPT_SOVITS_CPP_G2P_ZH_H
#define GPT_SOVITS_CPP_G2P_ZH_H

#include <GPTSoVITS/G2P/Base.h>

namespace GPTSoVITS::G2P {

class G2PZH : public IG2P {
protected:
  G2PRes _cleanText(const std::string &text) const override ;
};

};


#endif  // GPT_SOVITS_CPP_G2P_ZH_H

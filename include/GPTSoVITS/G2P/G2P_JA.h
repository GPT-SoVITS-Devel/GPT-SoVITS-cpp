//
// Created by 19254 on 2026/2/8.
//

#ifndef GSV_CPP_G2P_JA_H
#define GSV_CPP_G2P_JA_H

#include <GPTSoVITS/G2P/Base.h>

struct _Mecab;
typedef _Mecab Mecab;
struct _NJD;
typedef _NJD NJD;
struct _JPCommon;
typedef _JPCommon JPCommon;

namespace GPTSoVITS::G2P {

class G2PJA : public IG2P {
public:
  void WarmUp() override;
  G2PRes _cleanText(const std::string &text) const override;
};

class OpenJTalk {
  Mecab *m_mecab = nullptr;
  NJD *m_njd = nullptr;
  JPCommon *m_jpcommon = nullptr;

  void clear();

public:
  struct Feature {
    std::string string;
    std::string pos;
    std::string pos_group1;
    std::string pos_group2;
    std::string pos_group3;
    std::string ctype;
    std::string cform;
    std::string orig;
    std::string read;
    std::string pron;
    int acc;
    int mora_size;
    std::string chain_rule;
    int chain_flag;
  };

  OpenJTalk(const std::string &dict_path, const std::string &user_dict);
  ~OpenJTalk();

  std::vector<Feature> run_frontend(const std::string &text);
  std::vector<std::string> make_label(const std::vector<Feature> &features);

  /*
   *
   * @result: join: true 使用 string
   *   false 使用 vector
   * */
  std::pair<std::vector<std::string>,std::string> g2p(const std::string &text,bool kana=false,bool join=true);

  static std::unique_ptr<OpenJTalk> FromUserDict(const std::string &dict_path, const std::string &user_dict);
};

};// namespace GPTSovits::G2P


#endif  // GSV_CPP_G2P_JA_H

//
// Created by Huiyicc on 2026/1/17.
//

#ifndef GPT_SOVITS_CPP_GPTSOVITSCPP_H
#define GPT_SOVITS_CPP_GPTSOVITSCPP_H

#include <filesystem>
#include <map>
#include <string>
#include <utility>

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/G2P/Pipline.h"
#include "GPTSoVITS/model/tensor.h"
#include "model/ssl.h"
#include "model/vq.h"

namespace GPTSoVITS {

void SetGlobalResourcesPath(const std::string& path);
std::filesystem::path GetGlobalResourcesPath();
class GPTSoVITSPipline;

class SpeakerInfo {
  std::string m_speaker_name;
  std::string m_speaker_lang;
  std::unique_ptr<Model::Tensor> m_ssl_content;
  std::unique_ptr<AudioTools> m_speaker_16k;
  std::unique_ptr<AudioTools> m_speaker_32k;

public:
  friend GPTSoVITSPipline;

  std::string SpeakerName() { return m_speaker_name; }
  std::string SpeakerLang() { return m_speaker_lang; }
};

class GPTSoVITSPipline {
  std::map<std::string, SpeakerInfo> m_speaker_map;

public:
  explicit GPTSoVITSPipline(const std::shared_ptr<G2P::G2PPipline>& g2p_pipline,
                            const std::shared_ptr<Model::SSLModel>& ssl_model,
                            const std::shared_ptr<Model::VQModel>& vq_model)
      : m_g2p_pipline(g2p_pipline), m_ssl_model(ssl_model),m_vq_model(vq_model) {};
  virtual ~GPTSoVITSPipline() = default;

  const SpeakerInfo& CreateSpeaker(const std::string& speaker_name,
                                   const std::string& ref_audio_lang,
                                   const std::string& ref_audio_path,
                                   const std::string& ref_audio_text);

private:
  std::shared_ptr<G2P::G2PPipline> m_g2p_pipline;
  std::shared_ptr<Model::SSLModel> m_ssl_model;
  std::shared_ptr<Model::VQModel> m_vq_model;
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_GPTSOVITSCPP_H
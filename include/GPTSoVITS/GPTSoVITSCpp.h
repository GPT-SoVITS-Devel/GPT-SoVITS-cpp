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
#include "GPTSoVITS/Text/Sentence.h"
#include "GPTSoVITS/model/tensor.h"
#include "model/gpt_encoder.h"
#include "model/gpt_step.h"
#include "model/sovits.h"
#include "model/spectrogram.h"
#include "model/sv_embedding.h"
#include "model/ssl.h"
#include "model/vq.h"


namespace GPTSoVITS {

class _JsonImpl;
void SetGlobalResourcesPath(const std::string& path);
std::filesystem::path GetGlobalResourcesPath();
class GPTSoVITSPipline;

class SpeakerInfo {
  std::string m_speaker_name;
  std::string m_speaker_lang;
  std::shared_ptr<Bert::BertRes> m_bert_res;
  std::unique_ptr<Model::Tensor> m_ssl_content;
  std::unique_ptr<Model::Tensor> m_vq_codes;
  std::unique_ptr<AudioTools> m_speaker_16k;
  std::unique_ptr<AudioTools> m_speaker_32k;
  std::unique_ptr<Model::Tensor> m_refer_spec;
  std::unique_ptr<Model::Tensor> m_sv_emb;

public:
  friend GPTSoVITSPipline;

  std::string SpeakerName() { return m_speaker_name; }
  std::string SpeakerLang() { return m_speaker_lang; }
};

class GPTSoVITSPipline {
  std::map<std::string, SpeakerInfo> m_speaker_map;

public:
  explicit GPTSoVITSPipline(
      const std::string& config,
      const std::shared_ptr<G2P::G2PPipline>& g2p_pipline,
      const std::shared_ptr<Model::SSLModel>& ssl_model,
      const std::shared_ptr<Model::VQModel>& vq_model,
      const std::shared_ptr<Model::SpectrogramModel>& spectrogram_model,
      const std::shared_ptr<Model::SVEmbeddingModel>& sv_embedding_model,
      const std::shared_ptr<Model::GPTEncoderModel>& gpt_encoder_model,
      const std::shared_ptr<Model::GPTStepModel>& gpt_step_model,
      const std::shared_ptr<Model::SoVITSModel>& sovits_model);

  virtual ~GPTSoVITSPipline() = default;

  const SpeakerInfo& CreateSpeaker(const std::string& speaker_name,
                                   const std::string& ref_audio_lang,
                                   const std::filesystem::path& ref_audio_path,
                                   const std::string& ref_audio_text);

  std::unique_ptr<AudioTools> InferSpeaker(const std::string& speaker_name,
                                           const std::string& text,
                                           const std::string& text_lang = "zh",
                                           float temperature = 1.0f,
                                           float noise_scale = 0.5f,
                                           float speed = 1.0f);

private:
  std::shared_ptr<_JsonImpl> m_config;
  std::shared_ptr<G2P::G2PPipline> m_g2p_pipline;
  std::shared_ptr<Model::SSLModel> m_ssl_model;
  std::shared_ptr<Model::VQModel> m_vq_model;
  std::shared_ptr<Model::SpectrogramModel> m_spectrogram_model;
  std::shared_ptr<Model::SVEmbeddingModel> m_sv_embedding_model;
  std::shared_ptr<Model::GPTEncoderModel> m_gpt_encoder_model;
  std::shared_ptr<Model::GPTStepModel> m_gpt_step_model;
  std::shared_ptr<Model::SoVITSModel> m_sovits_model;

  // Helper methods
  static int64_t SampleTopK(const Model::Tensor* topk_values,
                            const Model::Tensor* topk_indices,
                            float temperature);
  static std::unique_ptr<Model::Tensor> ConcatTensor(const Model::Tensor* a,
                                                     const Model::Tensor* b,
                                                     int axis);
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_GPTSOVITSCPP_H
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
public:
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

  [[nodiscard]] std::string SpeakerName() const { return m_speaker_name; }
  [[nodiscard]] std::string SpeakerLang() const { return m_speaker_lang; }
  [[nodiscard]] std::shared_ptr<Bert::BertRes> BertRes() const { return m_bert_res; }

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

  /**
   * @brief 导出说话人数据到文件（用于云端创建-边缘推理）
   * @param speaker_name 说话人名称
   * @param output_path 输出文件路径
   * @param include_audio 是否包含音频数据（增加文件大小）
   * @return 是否成功
   */
  bool ExportSpeaker(const std::string& speaker_name,
                     const std::string& output_path,
                     bool include_audio = false);

  /**
   * @brief 从文件导入说话人数据（用于边缘设备加载）
   * @param input_path 输入文件路径
   * @param speaker_name 新的说话人名称（可选，如果不指定则使用文件中的名称）
   * @return 是否成功
   */
  bool ImportSpeaker(const std::string& input_path,
                     const std::string& speaker_name = "");

  /**
   * @brief 列出所有已创建的说话人
   * @return 说话人名称列表
   */
  std::vector<std::string> ListSpeakers() const;

  /**
   * @brief 移除说话人
   * @param speaker_name 说话人名称
   * @return 是否成功
   */
  bool RemoveSpeaker(const std::string& speaker_name);

  /**
   * @brief 检查说话人是否存在
   * @param speaker_name 说话人名称
   * @return 是否存在
   */
  bool HasSpeaker(const std::string& speaker_name) const;

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

  // 配置参数（从config.json读取）
  Model::DataType m_compute_precision = Model::DataType::kFloat32;  // 计算精度
  int m_sampling_rate = 32000;
  int m_max_len = 1000;  // GPT KV cache预分配最大长度
  int m_hop_length = 640;
  int m_filter_length = 2048;
  int m_mel_bins = 128;
  int m_sv_dim = 20480;  // SV embedding维度
  std::string m_model_version = "v2";

  // Helper methods
  static int64_t SampleTopK(const Model::Tensor* topk_values,
                            const Model::Tensor* topk_indices,
                            float temperature);
  static std::unique_ptr<Model::Tensor> ConcatTensor(const Model::Tensor* a,
                                                     const Model::Tensor* b,
                                                     int axis);

  // 初始化配置参数
  void InitializeConfig();
  // 检测模型精度
  void DetectModelPrecision();
  // 获取计算精度对应的数据类型
  Model::DataType GetComputeDataType() const;
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_GPTSOVITSCPP_H
//
// Created by iFlow CLI on 2026/2/20.
//

#ifndef GPT_SOVITS_CPP_EDGE_PIPELINE_H
#define GPT_SOVITS_CPP_EDGE_PIPELINE_H

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/G2P/Pipline.h"
#include "GPTSoVITS/Text/Sentence.h"
#include "GPTSoVITS/model/tensor.h"
#include "model/gpt_encoder.h"
#include "model/gpt_step.h"
#include "model/sovits.h"

namespace GPTSoVITS {

class _JsonImpl;
class SpeakerInfo;

/**
 * @brief 边缘推理 Pipeline
 *
 * 这是一个轻量级的 Pipeline，只加载推理所需的模型：
 * - BERT (用于目标文本处理)
 * - GPT Encoder (用于编码)
 * - GPT Step (用于生成)
 * - SoVITS (用于音频生成)
 *
 * 不需要的模型（仅 CreateSpeaker 需要）：
 * - SSL, VQ, Spectrogram, SV Embedding
 *
 * 说话人数据通过 ImportSpeaker 从打包文件加载。
 */
class EdgePipeline {
  std::map<std::string, SpeakerInfo> m_speaker_map;

public:
  /**
   * @brief 构造函数 - 只加载推理所需的模型
   * @param config 配置 JSON 字符串
   * @param model_path 模型文件目录路径
   * @param g2p_pipline G2P 管道（需要 BERT 处理器）
   * @param gpt_encoder_model GPT 编码器模型
   * @param gpt_step_model GPT 步进模型
   * @param sovits_model SoVITS 模型
   */
  EdgePipeline(
      const std::string& config,
      const std::string& model_path,
      const std::shared_ptr<G2P::G2PPipline>& g2p_pipline,
      const std::shared_ptr<Model::GPTEncoderModel>& gpt_encoder_model,
      const std::shared_ptr<Model::GPTStepModel>& gpt_step_model,
      const std::shared_ptr<Model::SoVITSModel>& sovits_model);

  virtual ~EdgePipeline() = default;

  /**
   * @brief 从文件导入说话人数据
   * @param input_path 说话人数据包路径
   * @param speaker_name 可选的新名称
   * @return 是否成功
   */
  bool ImportSpeaker(const std::string& input_path,
                     const std::string& speaker_name = "");

  /**
   * @brief 推理说话人（与 GPTSoVITSPipline 相同的接口）
   * @param speaker_name 说话人名称
   * @param text 目标文本
   * @param text_lang 文本语言
   * @param temperature 温度参数
   * @param noise_scale 噪声缩放
   * @param speed 速度
   * @return 生成的音频
   */
  std::unique_ptr<AudioTools> InferSpeaker(const std::string& speaker_name,
                                           const std::string& text,
                                           const std::string& text_lang = "zh",
                                           float temperature = 1.0f,
                                           float noise_scale = 0.5f,
                                           float speed = 1.0f);

  /**
   * @brief 列出所有已导入的说话人
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

  /**
   * @brief 获取模型信息
   * @return 模型信息字符串
   */
  std::string GetModelInfo() const;

private:
  std::shared_ptr<_JsonImpl> m_config;
  std::shared_ptr<G2P::G2PPipline> m_g2p_pipline;
  std::shared_ptr<Model::GPTEncoderModel> m_gpt_encoder_model;
  std::shared_ptr<Model::GPTStepModel> m_gpt_step_model;
  std::shared_ptr<Model::SoVITSModel> m_sovits_model;

  // 配置参数
  Model::DataType m_compute_precision = Model::DataType::kFloat32;
  int m_sampling_rate = 32000;
  int m_max_len = 1000;
  int m_hop_length = 640;
  int m_filter_length = 2048;
  int m_mel_bins = 128;
  int m_sv_dim = 20480;
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
  // 获取计算精度对应的数据类型
  Model::DataType GetComputeDataType() const;
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_EDGE_PIPELINE_H
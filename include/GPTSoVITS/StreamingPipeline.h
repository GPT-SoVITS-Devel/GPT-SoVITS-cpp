//
// Created by iFlow CLI on 2026/2/20.
//

#ifndef GPT_SOVITS_CPP_STREAMING_PIPELINE_H
#define GPT_SOVITS_CPP_STREAMING_PIPELINE_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/EdgePipeline.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS {

/**
 * @brief 流推理配置
 */
struct StreamingConfig {
  int chunk_length = 24;          // 分块长度（token 数）
  float pause_length = 0.3f;      // 段落间停顿（秒）
  int fade_length = 1280;         // 淡入淡出长度（采样点数）
  int h_len = 512;                // 历史长度（采样点数）
  int l_len = 16;                 // 前瞻长度（采样点数）
  bool enable_fade = true;        // 是否启用淡入淡出
  bool enable_mute_matrix = false; // 是否使用静音矩阵分割
};

/**
 * @brief 音频分块
 */
struct AudioChunk {
  std::vector<float> audio_data;  // 音频数据
  bool is_first;                  // 是否是第一个分块
  bool is_last;                   // 是否是最后一个分块
  int segment_index;              // 段落索引
  int chunk_index;                // 分块索引
  float duration;                 // 音频时长（秒）
};

/**
 * @brief 流推理 Pipeline
 *
 * 支持实时流式音频生成
 */
class StreamingPipeline {
public:
  /**
   * @brief 音频分块回调函数
   * @param chunk 音频分块
   */
  using AudioChunkCallback = std::function<void(const AudioChunk&)>;

  /**
   * @brief 构造函数
   * @param edge_pipeline 边缘推理 Pipeline
   * @param config 流推理配置
   */
  explicit StreamingPipeline(
      std::shared_ptr<EdgePipeline> edge_pipeline,
      const StreamingConfig& config = StreamingConfig());

  virtual ~StreamingPipeline() = default;

  /**
   * @brief 流式推理说话人
   * @param speaker_name 说话人名称
   * @param text 目标文本
   * @param text_lang 文本语言
   * @param callback 音频分块回调
   * @param temperature 温度参数
   * @param noise_scale 噪声缩放
   * @param speed 速度
   * @return 是否成功
   */
  bool InferSpeakerStreaming(
      const std::string& speaker_name,
      const std::string& text,
      const std::string& text_lang,
      AudioChunkCallback callback,
      float temperature = 1.0f,
      float noise_scale = 0.5f,
      float speed = 1.0f);

  /**
   * @brief 设置流推理配置
   * @param config 新的配置
   */
  void SetConfig(const StreamingConfig& config) { m_config = config; }

  /**
   * @brief 获取当前配置
   * @return 当前配置
   */
  const StreamingConfig& GetConfig() const { return m_config; }

private:
  std::shared_ptr<EdgePipeline> m_edge_pipeline;
  StreamingConfig m_config;

  /**
   * @brief 处理单个文本段落
   * @param speaker_info 说话人信息
   * @param segment 文本段落
   * @param segment_index 段落索引
   * @param callback 音频分块回调
   * @param temperature 温度参数
   * @param noise_scale 噪声缩放
   * @param speed 速度
   * @param prev_fade_out 前一个分块的淡出数据
   * @return 最后一个分块的淡出数据
   */
  std::vector<float> ProcessSegment(
      const SpeakerInfo& speaker_info,
      const std::string& segment,
      int segment_index,
      AudioChunkCallback callback,
      float temperature,
      float noise_scale,
      float speed,
      const std::vector<float>& prev_fade_out);

  /**
   * @brief 分割语义 tokens
   * @param semantic_tokens 语义 tokens
   * @param chunk_queue 分块队列
   * @return 分割后的分块数
   */
  int SplitSemanticTokens(
      const std::vector<int64_t>& semantic_tokens,
      std::vector<std::vector<int64_t>>& chunk_queue);

  /**
   * @brief 解码音频分块
   * @param chunk_tokens 分块 tokens
   * @param history_tokens 历史 tokens
   * @param lookahead_tokens 前瞻 tokens
   * @param speaker_info 说话人信息
   * @param noise_scale 噪声缩放
   * @param speed 速度
   * @return 解码后的音频数据
   */
  std::vector<float> DecodeChunk(
      const std::vector<int64_t>& chunk_tokens,
      const std::vector<int64_t>& history_tokens,
      const std::vector<int64_t>& lookahead_tokens,
      const SpeakerInfo& speaker_info,
      float noise_scale,
      float speed);

  /**
   * @brief 应用淡入淡出
   * @param audio 音频数据
   * @param fade_in 淡入数据
   * @param fade_out 淡出数据
   * @return 处理后的音频
   */
  std::vector<float> ApplyFade(
      const std::vector<float>& audio,
      const std::vector<float>& fade_in,
      const std::vector<float>& fade_out);

  /**
   * @brief 生成停顿音频
   * @param duration 停顿时长（秒）
   * @param sampling_rate 采样率
   * @return 停顿音频数据
   */
  std::vector<float> GeneratePause(
      float duration,
      int sampling_rate);
};

}  // namespace GPTSoVITS

#endif  // GPT_SOVITS_CPP_STREAMING_PIPELINE_H
//
// Created by iFlow CLI on 2026/2/20.
//

#include "GPTSoVITS/StreamingPipeline.h"
#include "boost/algorithm/string.hpp"
#include <algorithm>
#include <cmath>

#include "GPTSoVITS/Text/Sentence.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS {

StreamingPipeline::StreamingPipeline(
    std::shared_ptr<EdgePipeline> edge_pipeline,
    const StreamingConfig& config)
    : m_edge_pipeline(edge_pipeline), m_config(config) {
  PrintInfo("[StreamingPipeline] Initialized with config:");
  PrintInfo("  chunk_length: {}", m_config.chunk_length);
  PrintInfo("  pause_length: {}s", m_config.pause_length);
  PrintInfo("  fade_length: {}", m_config.fade_length);
}

bool StreamingPipeline::InferSpeakerStreaming(
    const std::string& speaker_name,
    const std::string& text,
    const std::string& text_lang,
    AudioChunkCallback callback,
    float temperature,
    float noise_scale,
    float speed) {
  if (!m_edge_pipeline->HasSpeaker(speaker_name)) {
    PrintError("[StreamingPipeline] Speaker '{}' not found", speaker_name);
    return false;
  }

  PrintInfo("[StreamingPipeline] Starting streaming inference for speaker: {}, text: {}",
            speaker_name, text);

  // 获取说话人信息
  // 注意：需要从 EdgePipeline 访问内部的 m_speaker_map
  // 这里简化处理，实际需要添加访问方法

  // 文本分句 - 使用 Sentence 类的流式处理方式
  std::vector<std::string> segments;
  Text::Sentence sentence(Text::Sentence::SentenceSplitMethod::Punctuation);
  
  sentence.AppendCallBack([&segments](const std::string& s) -> bool {
    segments.push_back(s);
    return true;
  });

  // 逐块添加文本（每次处理 11 个字符，参考 clean_text 示例）
  int chunk_size = 11;
  int index = 0;
  while (index < text.size()) {
    std::string chunk = text.substr(index, chunk_size);
    sentence.Append(chunk);
    index += chunk_size;
  }
  sentence.Flush();

  if (segments.empty()) {
    PrintWarn("[StreamingPipeline] No text segments to process");
    return false;
  }

  std::vector<float> prev_fade_out;

  // 遍历每个句子段落
  for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
    const std::string& segment = segments[seg_idx];

    PrintDebug("[StreamingPipeline] Processing segment {}/{}: {}",
              seg_idx + 1, segments.size(), segment);

    // 处理段落（TODO: 需要访问 speaker_info）
    // prev_fade_out = ProcessSegment(
    //     speaker_info, segment, seg_idx, callback,
    //     temperature, noise_scale, speed, prev_fade_out);

    // 添加段落间停顿
    if (seg_idx < segments.size() - 1 && m_config.pause_length > 0) {
      auto pause_audio = GeneratePause(m_config.pause_length, 32000);  // TODO: 从配置获取采样率

      AudioChunk pause_chunk;
      pause_chunk.audio_data = pause_audio;
      pause_chunk.is_first = false;
      pause_chunk.is_last = false;
      pause_chunk.segment_index = seg_idx;
      pause_chunk.chunk_index = -1;  // 停顿不是普通分块
      pause_chunk.duration = static_cast<float>(pause_audio.size()) / 32000;

      if (callback) {
        callback(pause_chunk);
      }

      // 重置淡出
      prev_fade_out.clear();
    }
  }

  PrintInfo("[StreamingPipeline] Streaming inference completed");
  return true;
}

std::vector<float> StreamingPipeline::ProcessSegment(
    const SpeakerInfo& speaker_info,
    const std::string& segment,
    int segment_index,
    AudioChunkCallback callback,
    float temperature,
    float noise_scale,
    float speed,
    const std::vector<float>& prev_fade_out) {
  // TODO: 实现完整的段落处理逻辑
  // 这需要访问 EdgePipeline 的内部模型和推理逻辑

  PrintWarn("[StreamingPipeline] ProcessSegment not fully implemented yet");

  // 返回空的淡出数据
  return {};
}

int StreamingPipeline::SplitSemanticTokens(
    const std::vector<int64_t>& semantic_tokens,
    std::vector<std::vector<int64_t>>& chunk_queue) {
  // TODO: 实现语义 token 分割逻辑
  // 参考 Python 的 mute_matrix 分割策略

  int chunk_index = 0;

  // 简单实现：按固定长度分割
  for (size_t i = 0; i < semantic_tokens.size(); i += m_config.chunk_length) {
    size_t end = std::min(i + m_config.chunk_length, semantic_tokens.size());
    std::vector<int64_t> chunk(semantic_tokens.begin() + i, semantic_tokens.begin() + end);
    chunk_queue.push_back(chunk);
    chunk_index++;
  }

  return chunk_index;
}

std::vector<float> StreamingPipeline::DecodeChunk(
    const std::vector<int64_t>& chunk_tokens,
    const std::vector<int64_t>& history_tokens,
    const std::vector<int64_t>& lookahead_tokens,
    const SpeakerInfo& speaker_info,
    float noise_scale,
    float speed) {
  // TODO: 实现音频分块解码
  // 这需要调用 SoVITS 模型进行推理

  PrintWarn("[StreamingPipeline] DecodeChunk not fully implemented yet");

  return {};
}

std::vector<float> StreamingPipeline::ApplyFade(
    const std::vector<float>& audio,
    const std::vector<float>& fade_in,
    const std::vector<float>& fade_out) {
  if (!m_config.enable_fade) {
    return audio;
  }

  std::vector<float> result = audio;

  // 应用淡入
  if (!fade_in.empty() && fade_in.size() <= audio.size()) {
    for (size_t i = 0; i < fade_in.size(); ++i) {
      result[i] = result[i] * (1.0f - fade_in[i]) + fade_in[i];
    }
  }

  // 应用淡出
  if (!fade_out.empty() && fade_out.size() <= audio.size()) {
    size_t start = audio.size() - fade_out.size();
    for (size_t i = 0; i < fade_out.size(); ++i) {
      result[start + i] = result[start + i] * fade_out[i];
    }
  }

  return result;
}

std::vector<float> StreamingPipeline::GeneratePause(
    float duration,
    int sampling_rate) {
  int num_samples = static_cast<int>(duration * sampling_rate);
  return std::vector<float>(num_samples, 0.0f);
}

}  // namespace GPTSoVITS
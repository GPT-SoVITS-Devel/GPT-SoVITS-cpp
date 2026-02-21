//
// Created by Huiyicc on 2026/1/17.
//
#include "GPTSoVITS/GPTSoVITSCpp.h"

#include <numeric>
#include <random>

#include "GPTSoVITS/plog.h"
#include "GPTSoVITS/Utils/speaker_serializer.h"
#include "nlohmann/json.hpp"

namespace GPTSoVITS {
std::filesystem::path g_globalResourcesPath =
    std::filesystem::current_path() / "res";

void SetGlobalResourcesPath(const std::string& path) {
  g_globalResourcesPath = path;
}
std::filesystem::path GetGlobalResourcesPath() { return g_globalResourcesPath; }

class _JsonImpl {
public:
  nlohmann::json data;
};

GPTSoVITSPipline::GPTSoVITSPipline(
    const std::string& config,
    const std::shared_ptr<G2P::G2PPipline>& g2p_pipline,
    const std::shared_ptr<Model::SSLModel>& ssl_model,
    const std::shared_ptr<Model::VQModel>& vq_model,
    const std::shared_ptr<Model::SpectrogramModel>& spectrogram_model,
    const std::shared_ptr<Model::SVEmbeddingModel>& sv_embedding_model,
    const std::shared_ptr<Model::GPTEncoderModel>& gpt_encoder_model,
    const std::shared_ptr<Model::GPTStepModel>& gpt_step_model,
    const std::shared_ptr<Model::SoVITSModel>& sovits_model)
    : m_g2p_pipline(g2p_pipline),
      m_ssl_model(ssl_model),
      m_vq_model(vq_model),
      m_spectrogram_model(spectrogram_model),
      m_sv_embedding_model(sv_embedding_model),
      m_gpt_encoder_model(gpt_encoder_model),
      m_gpt_step_model(gpt_step_model),
      m_sovits_model(sovits_model) {
  // 接口验证：确保所有必需的模型和方法都存在
#ifdef GSV_ENABLE_INTERFACE_VALIDATION
  using namespace Utils;

  // 注册模块依赖关系
  auto& dep_graph = ModuleDependencyGraph::Instance();
  dep_graph.RegisterModule("G2PPipline", {"BertModel"});
  dep_graph.RegisterModule("SSLModel", {});
  dep_graph.RegisterModule("VQModel", {"SSLModel"});
  dep_graph.RegisterModule("SpectrogramModel", {});
  dep_graph.RegisterModule("SVEmbeddingModel", {});
  dep_graph.RegisterModule("GPTEncoderModel", {"BertModel"});
  dep_graph.RegisterModule("GPTStepModel", {"GPTEncoderModel"});
  dep_graph.RegisterModule("SoVITSModel", {"VQModel", "SpectrogramModel", "SVEmbeddingModel"});
  dep_graph.RegisterModule("GPTSoVITSPipline",
                          {"G2PPipline", "SSLModel", "VQModel", "SpectrogramModel",
                           "SVEmbeddingModel", "GPTEncoderModel", "GPTStepModel",
                           "SoVITSModel"});

  // 验证依赖关系
  if (!dep_graph.ValidateDependencies()) {
    PrintError("[GPTSoVITSPipline] Module dependency validation failed!");
  }

  // 验证所有接口
  if (!InterfaceValidator::ValidatePipelineInterface<GPTSoVITSPipline>()) {
    PrintError("[GPTSoVITSPipline] Pipeline interface validation failed!");
  }
#endif

  m_config = std::make_shared<_JsonImpl>();
  m_config->data = nlohmann::json::parse(config);

  // 初始化配置参数
  InitializeConfig();

  // 检测模型精度
  DetectModelPrecision();

  PrintDebug("GPT-SoVITS Pipeline initialized:");
  PrintDebug("  Model version: {}", m_model_version);
  PrintDebug("  Sampling rate: {} Hz", m_sampling_rate);
  PrintDebug("  Max sequence length: {}", m_max_len);
  PrintDebug("  Compute precision: {}",
            m_compute_precision == Model::DataType::kFloat16 ? "FP16" : "FP32");
  PrintDebug("  SV embedding dim: {}", m_sv_dim);
}

const SpeakerInfo& GPTSoVITSPipline::CreateSpeaker(
    const std::string& speaker_name, const std::string& ref_audio_lang,
    const std::filesystem::path& ref_audio_path,
    const std::string& ref_audio_text) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter != m_speaker_map.end()) {
    return iter->second;
  }

  PrintDebug("Creating new speaker: {}", speaker_name);

  SpeakerInfo info;
  info.m_speaker_name = speaker_name;
  info.m_speaker_lang = ref_audio_lang;

  auto refAudio = AudioTools::FromFile(ref_audio_path.string());

  auto audio16k = refAudio->ReSample(16000);
  auto samples16k_raw = audio16k->ReadSamples();
  
  // 给16k音频增加0.3s静音填充, 仅用于 SSL/VQ 提取, 匹配Python端行为
  auto samples16k_padded = samples16k_raw;
  size_t padding_size = static_cast<size_t>(16000 * 0.3);
  samples16k_padded.insert(samples16k_padded.end(), padding_size, 0.0f);
  
  info.m_speaker_16k = AudioTools::FromByte(samples16k_padded, 16000);

  // 32k (or native sr) for SoVITS
  info.m_speaker_32k = refAudio->ReSample(32000);

  // bert embeddings
  info.m_bert_res =
      m_g2p_pipline->GetPhoneAndBert(ref_audio_text, ref_audio_lang);

  // get ssl and vq codes
  info.m_ssl_content =
      m_ssl_model->GetSSLContent(info.m_speaker_16k->ReadSamples());
  auto vq_raw = m_vq_model->GetVQCodes(info.m_ssl_content.get());
  if (!vq_raw) {
    THROW_ERRORN("Failed to get VQ codes for speaker {}", speaker_name);
  }

  // prompt_semantic = codes[0, 0][None, :]
  if (vq_raw->Shape().size() == 3) {
    auto shape = vq_raw->Shape();
    int64_t B = shape[0];
    int64_t K = shape[1];
    int64_t T = shape[2];
    PrintInfo("  VQ raw shape: [{}, {}, {}]", B, K, T);
    
    auto sliced = Model::Tensor::Empty({1, T}, Model::DataType::kInt64, Model::DeviceType::kCPU);
    auto vq_cpu = vq_raw->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
    std::memcpy(sliced->Data<int64_t>(), vq_cpu->Data<int64_t>(), T * sizeof(int64_t));
    info.m_vq_codes = std::move(sliced);
  } else if (vq_raw->Shape().size() == 2 && vq_raw->Shape()[0] > 1) {
    auto T = vq_raw->Shape()[1];
    auto sliced = Model::Tensor::Empty({1, T}, Model::DataType::kInt64, Model::DeviceType::kCPU);
    auto vq_cpu = vq_raw->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
    std::memcpy(sliced->Data<int64_t>(), vq_cpu->Data<int64_t>(), T * sizeof(int64_t));
    info.m_vq_codes = std::move(sliced);
  } else {
    info.m_vq_codes = std::move(vq_raw);
    if (info.m_vq_codes->Shape().size() == 1) {
      info.m_vq_codes->Reshape({1, info.m_vq_codes->Shape()[0]});
    }
  }

  PrintDebug("  Final VQ codes shape: [{}, {}]",
            info.m_vq_codes->Shape()[0], info.m_vq_codes->Shape()[1]);

  auto sr = m_config->data["data"].value<int>("sampling_rate", 32000);

  info.m_refer_spec = m_spectrogram_model->ComputeSpec(
      info.m_speaker_32k->ReSample(sr)->ReadSamples());
  if (info.m_refer_spec->Shape().size() == 3) {
    info.m_refer_spec->Reshape(
        {info.m_refer_spec->Shape()[1], info.m_refer_spec->Shape()[2]});
  }

  // SV embedding
  info.m_sv_emb =
      m_sv_embedding_model->ComputeEmbedding(samples16k_raw);
  if (info.m_sv_emb->Shape().size() == 2) {
    info.m_sv_emb->Reshape({info.m_sv_emb->Shape()[1]});
  }

  m_speaker_map[speaker_name] = std::move(info);
  return m_speaker_map[speaker_name];
}

std::unique_ptr<AudioTools> GPTSoVITSPipline::InferSpeaker(
    const std::string& speaker_name, const std::string& text,
    const std::string& text_lang, float temperature, float noise_scale,
    float speed) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter == m_speaker_map.end()) {
    PrintError("Speaker not found: {}", speaker_name);
    return nullptr;
  }

  const auto& speaker_info = iter->second;

  PrintDebug("Starting inference for speaker: {}", speaker_name);

  // Use Text::Sentence for multi-language text splitting
  Text::Sentence sentence(Text::Sentence::SentenceSplitMethod::Punctuation);

  std::vector<std::string> segments;
  std::vector<std::shared_ptr<Bert::BertRes>> segment_bert_res;

  sentence.AppendCallBack([this, &segments, &segment_bert_res,
                           &text_lang](const std::string& seg) -> bool {
    PrintDebug(">>> [Segment Split] Processing: {}", seg);
    segments.push_back(seg);
    try {
      // Get phones and bert for target text
      auto target_bert_res = m_g2p_pipline->GetPhoneAndBert(seg, text_lang);
      segment_bert_res.push_back(target_bert_res);
    } catch (const std::exception& e) {
      PrintError("    [Inference Error] Failed to process segment: {}",
                 e.what());
    }
    return true;
  });

  sentence.Append(text);
  sentence.Flush();

  if (segments.empty()) {
    PrintWarn("No text segments to process");
    return nullptr;
  }

  PrintDebug("Processing {} text segments", segments.size());

  std::vector<float> final_audio;

  // Get reference text phones and bert features
  auto& ref_phones = speaker_info.m_bert_res->PhoneSeq;
  auto& ref_bert = speaker_info.m_bert_res->BertSeq;

  // Process each segment
  for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
    PrintInfo("Processing segment {}/{}: {}", seg_idx + 1, segments.size(),
              segments[seg_idx]);

    auto& target_bert_res = segment_bert_res[seg_idx];
    auto& target_phones = target_bert_res->PhoneSeq;
    auto& target_bert = target_bert_res->BertSeq;

    // Concatenate reference and target
    auto all_phones = ConcatTensor(ref_phones.get(), target_phones.get(), 0);
    auto all_bert = ConcatTensor(ref_bert.get(), target_bert.get(), 1);

    PrintDebug("  ref_bert shape: [{}, {}, {}]",
              ref_bert->Shape().size() > 0 ? ref_bert->Shape()[0] : 0,
              ref_bert->Shape().size() > 1 ? ref_bert->Shape()[1] : 0,
              ref_bert->Shape().size() > 2 ? ref_bert->Shape()[2] : 0);
    PrintDebug("  target_bert shape: [{}, {}, {}]",
              target_bert->Shape().size() > 0 ? target_bert->Shape()[0] : 0,
              target_bert->Shape().size() > 1 ? target_bert->Shape()[1] : 0,
              target_bert->Shape().size() > 2 ? target_bert->Shape()[2] : 0);
    PrintDebug("  all_bert (before reshape) shape: [{}, {}, {}]",
              all_bert->Shape().size() > 0 ? all_bert->Shape()[0] : 0,
              all_bert->Shape().size() > 1 ? all_bert->Shape()[1] : 0,
              all_bert->Shape().size() > 2 ? all_bert->Shape()[2] : 0);

    if (all_bert->Shape().size() == 2) {
      all_bert->Reshape({1, all_bert->Shape()[0], all_bert->Shape()[1]});
      PrintDebug("  Reshaped all_bert to: [{}, {}, {}]",
                all_bert->Shape()[0], all_bert->Shape()[1], all_bert->Shape()[2]);
    }

    // 准备输入时使用正确的数据类型
    auto compute_dtype = GetComputeDataType();

    if (all_bert->Type() != compute_dtype) {
      PrintDebug("  Converting bert_feature from {} to {} for GPT Encoder",
                all_bert->Type() == Model::DataType::kFloat32 ? "float32" : "float16",
                compute_dtype == Model::DataType::kFloat32 ? "float32" : "float16");
      all_bert = all_bert->To(all_bert->GetDevice(), compute_dtype);
    }

    // phoneme_ids (1, seq_len)
    auto phoneme_ids = all_phones->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
    if (phoneme_ids->Shape().size() == 1) {
      phoneme_ids->Reshape({1, phoneme_ids->Shape()[0]});
    }

    // prompts (1, prompt_len)
    auto prompts = speaker_info.m_vq_codes->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
    if (prompts->Shape().size() == 1) {
      prompts->Reshape({1, prompts->Shape()[0]});
    }

    PrintDebug("  GPT Encoder inputs:");
    PrintDebug("    phoneme_ids shape: [{}, {}], dtype: {}",
              phoneme_ids->Shape()[0], phoneme_ids->Shape()[1],
              phoneme_ids->Type() == Model::DataType::kInt64 ? "int64" : "other");
    if (phoneme_ids->ElementCount() > 5) {
      auto* p_ids = phoneme_ids->Data<int64_t>();
      PrintDebug("    phoneme_ids[0..4]: {}, {}, {}, {}, {}", p_ids[0], p_ids[1], p_ids[2], p_ids[3], p_ids[4]);
    }
    PrintDebug("    prompts shape: [{}, {}], dtype: {}",
              prompts->Shape()[0], prompts->Shape()[1],
              prompts->Type() == Model::DataType::kInt64 ? "int64" : "other");
    if (prompts->ElementCount() > 5) {
      auto* p_prompts = prompts->Data<int64_t>();
      PrintDebug("    prompts[0..4]: {}, {}, {}, {}, {}", p_prompts[0], p_prompts[1], p_prompts[2], p_prompts[3], p_prompts[4]);
    }
    PrintDebug("    bert_feature shape: [{}, {}, {}], dtype: {}",
              all_bert->Shape()[0], all_bert->Shape()[1], all_bert->Shape()[2],
              all_bert->Type() == Model::DataType::kFloat16 ? "float16" : "float32");

    // Run GPT Encoder
    auto encoder_output = m_gpt_encoder_model->Encode(
        phoneme_ids.get(), prompts.get(), all_bert.get());

    if (encoder_output.topk_values && encoder_output.topk_values->ElementCount() > 0) {
      auto topk_values_cpu = encoder_output.topk_values->To(
          Model::Device(Model::DeviceType::kCPU), Model::DataType::kFloat32);
      const float* enc_values = topk_values_cpu->Data<float>();
      int enc_k = topk_values_cpu->ElementCount();
      
      PrintDebug("  GPT Encoder topk_values[0..5]: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}",
                enc_values[0], enc_values[1], enc_values[2], enc_values[3], enc_values[4]);

      bool enc_has_nan = false;
      for (int j = 0; j < enc_k; ++j) {
        if (!std::isfinite(enc_values[j])) {
          enc_has_nan = true;
          break;
        }
      }
      if (enc_has_nan) {
        PrintError("GPT Encoder topk_values contains NaN/Inf!");
      }
    }

    // Sample first token
    int64_t first_token =
        SampleTopK(encoder_output.topk_values.get(),
                   encoder_output.topk_indices.get(), temperature);

    PrintDebug("  First sampled token: {}", first_token);

    // Prepare semantic list
    std::vector<std::unique_ptr<Model::Tensor>> decoded_semantic_list;
    decoded_semantic_list.push_back(speaker_info.m_vq_codes->Clone());

    auto first_token_tensor = Model::Tensor::Empty(
        {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
    first_token_tensor->At<int64_t>(0) = first_token;

    // GPT Step loop
    auto current_samples = first_token_tensor->Clone();

    decoded_semantic_list.push_back(std::move(first_token_tensor));
    auto k_cache = std::move(encoder_output.k_cache);
    auto v_cache = std::move(encoder_output.v_cache);
    auto x_len = std::move(encoder_output.x_len);
    auto y_len = std::move(encoder_output.y_len);

    int max_steps = 1500;
    int64_t eos_token = 1024;
    int steps = 0;
    int consecutive_invalid_count = 0;  // 连续无效输出计数
    const int max_consecutive_invalid = 10;  // 允许的最大连续无效次数

    for (int i = 0; i < max_steps; ++i) {
      auto idx_tensor = Model::Tensor::Empty({1}, Model::DataType::kInt64,
                                             Model::DeviceType::kCPU);
      idx_tensor->At<int64_t>(0) = i;

      if (i == 0) {
        PrintDebug("  GPT Step #0: idx={}, current_samples={}",
                  idx_tensor->At<int64_t>(0), current_samples->At<int64_t>(0));
        if (x_len && x_len->ElementCount() > 0) {
          PrintDebug("  x_len={}", x_len->At<int64_t>(0));
        }
        if (y_len && y_len->ElementCount() > 0) {
          PrintDebug("  y_len={}", y_len->At<int64_t>(0));
        }
      }

      auto step_output = m_gpt_step_model->Step(
          current_samples.get(), k_cache.get(), v_cache.get(), idx_tensor.get(),
          x_len.get(), y_len.get());

      // 检查输出是否有效
      bool output_valid = true;
      if (!step_output.topk_values || step_output.topk_values->ElementCount() == 0) {
        PrintError("GPT Step {}: topk_values is empty!", i);
        output_valid = false;
      } else {
        // 确保数据在CPU上且为Float32，用于检查
        auto topk_values_cpu = step_output.topk_values->To(
            Model::Device(Model::DeviceType::kCPU), Model::DataType::kFloat32);
        const float* values = topk_values_cpu->Data<float>();
        int k = topk_values_cpu->ElementCount();
        bool has_nan = false;
        for (int j = 0; j < k; ++j) {
          if (!std::isfinite(values[j])) {
            has_nan = true;
            break;
          }
        }
        if (has_nan) {
          PrintError("GPT Step {}: topk_values contains NaN/Inf!", i);
          output_valid = false;
        }
      }

      if (!output_valid) {
        consecutive_invalid_count++;
        if (consecutive_invalid_count >= max_consecutive_invalid) {
          PrintError("GPT Step failed {} times consecutively, terminating generation at step {}", consecutive_invalid_count, i);
          break;
        }
        // 使用上一次的token继续
        int64_t last_valid_token = current_samples->At<int64_t>(0);
        auto next_token_tensor = Model::Tensor::Empty(
            {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
        next_token_tensor->At<int64_t>(0) = last_valid_token;
        current_samples = next_token_tensor->Clone();
        decoded_semantic_list.push_back(std::move(next_token_tensor));
        steps++;
        continue;
      }

      consecutive_invalid_count = 0;  // 重置计数器

      // Update cache
      k_cache = std::move(step_output.k_cache_new);
      v_cache = std::move(step_output.v_cache_new);

      // Sample next token
      int64_t next_token =
          SampleTopK(step_output.topk_values.get(),
                     step_output.topk_indices.get(), temperature);

      // 检查token有效性
      if (next_token < 0 || next_token >= 1025) {
        PrintWarn("GPT Step {}: Generated invalid token {}, clamping to valid range", i, next_token);
        next_token = std::max<int64_t>(0, std::min<int64_t>(next_token, 1024));
      }

      auto next_token_tensor = Model::Tensor::Empty(
          {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
      next_token_tensor->At<int64_t>(0) = next_token;

      // 先clone用于下一次step的输入
      current_samples = next_token_tensor->Clone();
      // 然后move到语义列表
      decoded_semantic_list.push_back(std::move(next_token_tensor));

      steps++;

      // Check for EOS
      if (next_token == eos_token) {
        PrintInfo("Generated {} tokens before EOS", steps);
        break;
      }

      // 定期打印进度
      if (steps % 100 == 0) {
        PrintDebug("  GPT generation progress: {} tokens generated", steps);
      }
    }

    if (steps >= max_steps) {
      PrintWarn("GPT generation reached max_steps ({}) without EOS", max_steps);
    }

    // Concatenate all semantic tokens (确保全在CPU上，以便处理)
    std::vector<std::unique_ptr<Model::Tensor>> semantic_cpu_list;
    std::vector<Model::Tensor*> semantic_ptrs;
    for (const auto& s : decoded_semantic_list) {
      semantic_cpu_list.push_back(s->ToCPU());
      semantic_ptrs.push_back(semantic_cpu_list.back().get());
    }
    auto pred_semantic = Model::Tensor::Concat(semantic_ptrs, 1);

    // Remove prompt part
    int prompt_len = speaker_info.m_vq_codes->Shape()[1];
    auto generated_sem =
        Model::Tensor::Empty({1, 1, pred_semantic->Shape()[1] - prompt_len},
                             Model::DataType::kInt64, Model::DeviceType::kCPU);

    auto pred_semantic_data = pred_semantic->Data<int64_t>();
    auto generated_sem_data = generated_sem->Data<int64_t>();
    std::memcpy(generated_sem_data, pred_semantic_data + prompt_len,
                (pred_semantic->Shape()[1] - prompt_len) * sizeof(int64_t));

    // Remove trailing EOS
    if (generated_sem->Shape()[2] > 0) {
      int64_t last_token =
          generated_sem->At<int64_t>(generated_sem->Shape()[2] - 1);
      if (last_token == eos_token) {
        auto trimmed_sem = Model::Tensor::Empty(
            {1, 1, generated_sem->Shape()[2] - 1}, Model::DataType::kInt64,
            Model::DeviceType::kCPU);
        std::memcpy(trimmed_sem->Data<int64_t>(),
                    generated_sem->Data<int64_t>(),
                    (generated_sem->Shape()[2] - 1) * sizeof(int64_t));
        generated_sem = std::move(trimmed_sem);
      }
    }

    // SoVITS解码
    // (1, seq_len)
    auto text_seq_tensor = target_phones->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kInt64);
    if (text_seq_tensor->Shape().size() == 1) {
      text_seq_tensor->Reshape({1, text_seq_tensor->Shape()[0]});
    }

    // 参考谱图
    auto refer_spec_reshaped = speaker_info.m_refer_spec->Clone();
    if (refer_spec_reshaped->Shape().size() == 2) {
      refer_spec_reshaped->Reshape({1, refer_spec_reshaped->Shape()[0],
                                    refer_spec_reshaped->Shape()[1]});
    }

    // SV embedding
    auto sv_emb_reshaped = speaker_info.m_sv_emb->Clone();
    if (sv_emb_reshaped->Shape().size() == 1) {
      sv_emb_reshaped->Reshape({1, sv_emb_reshaped->Shape()[0]});
    }

    // 生成音频
    PrintDebug("  SoVITS decoding...");
    auto audio_output = m_sovits_model->GenerateTensor(
        generated_sem.get(), text_seq_tensor.get(), refer_spec_reshaped.get(),
        sv_emb_reshaped.get(), noise_scale, speed);

    if (audio_output && audio_output->ElementCount() > 0) {
      auto audio_cpu = audio_output->To(Model::Device(Model::DeviceType::kCPU), Model::DataType::kFloat32);
      const float* audio_data = audio_cpu->Data<float>();
      size_t audio_len = audio_cpu->ElementCount();

      // 去除DC偏移（每个segment单独处理，防止漂移）
      std::vector<float> segment_audio(audio_data, audio_data + audio_len);
      float mean_val =
          std::accumulate(segment_audio.begin(), segment_audio.end(), 0.0f) /
          audio_len;
      for (auto& sample : segment_audio) {
        sample -= mean_val;
      }

      // 添加到最终音频
      final_audio.insert(final_audio.end(), segment_audio.begin(),
                         segment_audio.end());
      PrintDebug("  Generated {} audio samples for segment {}", audio_len,
                seg_idx + 1);
    } else {
      PrintWarn("  SoVITS generated empty audio for segment {}", seg_idx + 1);
    }

    // 添加停顿
    if (seg_idx < segments.size() - 1) {
      int pause_samples = static_cast<int>(m_sampling_rate * 0.3f);
      final_audio.insert(final_audio.end(), pause_samples, 0.0f);
    }
  }

  // 全局峰值归一化
  if (!final_audio.empty()) {
    // 找到最大幅度
    float max_amp = 0.0f;
    for (const auto& sample : final_audio) {
      float abs_sample = std::abs(sample);
      if (abs_sample > max_amp) {
        max_amp = abs_sample;
      }
    }

    // 归一化到0.9倍峰值
    if (max_amp > 1e-5f) {
      float scale = 0.9f / max_amp;
      for (auto& sample : final_audio) {
        sample *= scale;
      }
    }

    PrintDebug("Final audio: {} samples ({} seconds), peak amplitude: {:.4f}",
              final_audio.size(),
              static_cast<float>(final_audio.size()) / m_sampling_rate,
              max_amp);

    return AudioTools::FromByte(final_audio,m_sampling_rate);
  } else {
    PrintWarn("Generated empty audio");
    return AudioTools::FromEmpty(m_sampling_rate);
  }
}

namespace {

// Helper function to sample from top-k
int64_t sample_top_k(const Model::Tensor* topk_values,
                     const Model::Tensor* topk_indices, float temperature) {
  if (!topk_values || !topk_indices || topk_values->ElementCount() == 0) {
    return 0;
  }

  // 确保数据在CPU上且类型正确，用于采样
  auto values_cpu = topk_values->To(Model::Device(Model::DeviceType::kCPU),
                                     Model::DataType::kFloat32);
  auto indices_cpu = topk_indices->To(Model::Device(Model::DeviceType::kCPU),
                                       Model::DataType::kInt64);

  auto values = values_cpu->Data<float>();
  auto indices = indices_cpu->Data<int64_t>();
  int k = values_cpu->ElementCount();

  if (k == 0) {
    PrintError("sample_top_k: k is zero!");
    return 0;
  }

  // Apply temperature
  std::vector<float> probs(values, values + k);
  if (temperature != 1.0f && temperature > 1e-6f) {
    for (auto& p : probs) {
      p /= temperature;
    }
  }

  // Find max for numerical stability
  float max_val = *std::max_element(probs.begin(), probs.end());

  // Apply softmax with numerical stability checks
  float sum = 0.0f;
  for (size_t i = 0; i < probs.size(); ++i) {
    probs[i] -= max_val;  // Subtract max to avoid overflow

    // Clamp to prevent exp overflow/underflow
    if (probs[i] > 50.0f) {
      probs[i] = 50.0f;
    } else if (probs[i] < -50.0f) {
      probs[i] = -50.0f;
    }

    probs[i] = std::exp(probs[i]);

    // Check for NaN or Inf
    if (!std::isfinite(probs[i])) {
      PrintWarn("sample_top_k: Invalid exp value at index {}, resetting to 0", i);
      probs[i] = 0.0f;
    }

    sum += probs[i];
  }

  // Normalize probabilities
  if (sum > 1e-10f) {
    for (auto& p : probs) {
      p /= sum;
    }
  } else {
    // Fallback: uniform distribution if sum is too small
    PrintWarn("sample_top_k: Sum of probabilities is too small ({:.6f}), using uniform distribution", sum);
    float uniform_prob = 1.0f / static_cast<float>(k);
    for (auto& p : probs) {
      p = uniform_prob;
    }
  }

  // Validate probabilities
  for (const auto& p : probs) {
    if (p < 0.0f || !std::isfinite(p)) {
      PrintError("sample_top_k: Invalid probability detected, falling back to argmax");
      // Fallback to argmax
      int max_idx = 0;
      float max_prob = probs[0];
      for (int i = 1; i < k; ++i) {
        if (probs[i] > max_prob) {
          max_prob = probs[i];
          max_idx = i;
        }
      }
      return indices[max_idx];
    }
  }

  // Sample using discrete distribution
  try {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int choice = dist(gen);
    return indices[choice];
  } catch (const std::exception& e) {
    PrintError("sample_top_k: discrete_distribution failed: {}, falling back to argmax", e.what());
    // Fallback to argmax
    int max_idx = 0;
    float max_prob = probs[0];
    for (int i = 1; i < k; ++i) {
      if (probs[i] > max_prob) {
        max_prob = probs[i];
        max_idx = i;
      }
    }
    return indices[max_idx];
  }
}

}  // anonymous namespace
int64_t GPTSoVITSPipline::SampleTopK(const Model::Tensor* topk_values,
                                     const Model::Tensor* topk_indices,
                                     float temperature) {
  return sample_top_k(topk_values, topk_indices, temperature);
}

std::unique_ptr<Model::Tensor> GPTSoVITSPipline::ConcatTensor(
    const Model::Tensor* a, const Model::Tensor* b, int axis) {
  std::vector<Model::Tensor*> tensors = {const_cast<Model::Tensor*>(a),
                                         const_cast<Model::Tensor*>(b)};
  return Model::Tensor::Concat(tensors, axis);
}

void GPTSoVITSPipline::InitializeConfig() {
  // 从config.json读取基本参数
  if (m_config->data.contains("data")) {
    auto& data = m_config->data["data"];
    m_sampling_rate = data.value<int>("sampling_rate", 32000);
    m_max_len = data.value<int>("max_len", 1000);
    m_hop_length = data.value<int>("hop_length", 640);
    m_filter_length = data.value<int>("filter_length", 2048);
  }

  if (m_config->data.contains("model")) {
    auto& model = m_config->data["model"];
    m_model_version = model.value<std::string>("version", "v2");
  }

  if (m_config->data.contains("sv_embedding")) {
    auto& sv_emb = m_config->data["sv_embedding"];
    m_sv_dim = sv_emb.value<int>("embedding_size", 20480);
  }

  // 更新SoVITS模型的SV维度
  if (m_sovits_model) {
    m_sovits_model->SetSVDim(m_sv_dim);
  }
}

void GPTSoVITSPipline::DetectModelPrecision() {
  // 通过检查GPT Encoder的输入数据类型来检测模型精度
  if (m_gpt_encoder_model && m_gpt_encoder_model->GetModel()) {
    auto dtype =
        m_gpt_encoder_model->GetModel()->GetInputDataType("bert_feature");
    if (dtype == Model::DataType::kFloat16) {
      m_compute_precision = Model::DataType::kFloat16;
      PrintInfo("Detected FP16 model from GPT Encoder input");
    } else {
      m_compute_precision = Model::DataType::kFloat32;
      PrintInfo("Detected FP32 model from GPT Encoder input");
    }
  } else {
    PrintWarn("Failed to detect model precision, defaulting to FP32");
    m_compute_precision = Model::DataType::kFloat32;
  }
}

Model::DataType GPTSoVITSPipline::GetComputeDataType() const {
  return m_compute_precision;
}

// ============ 说话人数据导入/导出方法 ============

bool GPTSoVITSPipline::ExportSpeaker(const std::string& speaker_name,
                                     const std::string& output_path,
                                     bool include_audio) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter == m_speaker_map.end()) {
    PrintError("[GPTSoVITSPipline] Speaker '{}' not found", speaker_name);
    return false;
  }

  PrintInfo("[GPTSoVITSPipline] Exporting speaker '{}' to: {}",
            speaker_name, output_path);

  bool success = Utils::SpeakerSerializer::SerializeToFile(
      iter->second, output_path, include_audio);

  if (success) {
    auto package_size = Utils::SpeakerSerializer::GetPackageSize(output_path);
    PrintInfo("[GPTSoVITSPipline] Successfully exported speaker '{}', package size: {} bytes",
              speaker_name, package_size);
  }

  return success;
}

bool GPTSoVITSPipline::ImportSpeaker(const std::string& input_path,
                                     const std::string& speaker_name) {
  PrintInfo("[GPTSoVITSPipline] Importing speaker from: {}", input_path);

  // 验证数据包
  if (!Utils::SpeakerSerializer::ValidatePackage(input_path)) {
    PrintError("[GPTSoVITSPipline] Invalid speaker package: {}", input_path);
    return false;
  }

  // 获取数据包信息
  auto package_info = Utils::SpeakerSerializer::GetPackageInfo(input_path);
  if (!package_info) {
    PrintError("[GPTSoVITSPipline] Failed to read package info: {}", input_path);
    return false;
  }

  // 确定说话人名称
  std::string final_name = speaker_name.empty() ?
      package_info->speaker_name : speaker_name;

  // 检查是否已存在
  if (m_speaker_map.find(final_name) != m_speaker_map.end()) {
    PrintWarn("[GPTSoVITSPipline] Speaker '{}' already exists, will be overwritten",
             final_name);
  }

  // 反序列化
  auto speaker_info = Utils::SpeakerSerializer::DeserializeFromFile(input_path);
  if (!speaker_info) {
    PrintError("[GPTSoVITSPipline] Failed to deserialize speaker from: {}", input_path);
    return false;
  }

  // 更新说话人名称
  speaker_info->m_speaker_name = final_name;

  // 存储到映射表
  m_speaker_map[final_name] = std::move(*speaker_info);

  PrintInfo("[GPTSoVITSPipline] Successfully imported speaker '{}', lang: {}",
            final_name, speaker_info->m_speaker_lang);

  return true;
}

std::vector<std::string> GPTSoVITSPipline::ListSpeakers() const {
  std::vector<std::string> speaker_names;
  speaker_names.reserve(m_speaker_map.size());

  for (const auto& [name, _] : m_speaker_map) {
    speaker_names.push_back(name);
  }

  return speaker_names;
}

bool GPTSoVITSPipline::RemoveSpeaker(const std::string& speaker_name) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter == m_speaker_map.end()) {
    PrintError("[GPTSoVITSPipline] Speaker '{}' not found", speaker_name);
    return false;
  }

  m_speaker_map.erase(iter);
  PrintInfo("[GPTSoVITSPipline] Removed speaker: {}", speaker_name);

  return true;
}

bool GPTSoVITSPipline::HasSpeaker(const std::string& speaker_name) const {
  return m_speaker_map.find(speaker_name) != m_speaker_map.end();
}

}  // namespace GPTSoVITS
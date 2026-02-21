//
// Created by iFlow CLI on 2026/2/20.
//

#include "GPTSoVITS/EdgePipeline.h"

#include <numeric>
#include <random>

#include "GPTSoVITS/plog.h"
#include "GPTSoVITS/Utils/speaker_serializer.h"
#include "nlohmann/json.hpp"

namespace GPTSoVITS {

class _JsonImpl {
public:
  nlohmann::json data;
};

EdgePipeline::EdgePipeline(
    const std::string& config,
    const std::string& model_path,
    const std::shared_ptr<G2P::G2PPipline>& g2p_pipline,
    const std::shared_ptr<Model::GPTEncoderModel>& gpt_encoder_model,
    const std::shared_ptr<Model::GPTStepModel>& gpt_step_model,
    const std::shared_ptr<Model::SoVITSModel>& sovits_model)
    : m_g2p_pipline(g2p_pipline),
      m_gpt_encoder_model(gpt_encoder_model),
      m_gpt_step_model(gpt_step_model),
      m_sovits_model(sovits_model) {
  m_config = std::make_shared<_JsonImpl>();
  m_config->data = nlohmann::json::parse(config);

  // 初始化配置参数
  InitializeConfig();

  PrintInfo("[EdgePipeline] Initialized with minimum model set:");
  PrintInfo("  Model path: {}", model_path);
  PrintInfo("  Model version: {}", m_model_version);
  PrintInfo("  Sampling rate: {} Hz", m_sampling_rate);
  PrintInfo("  Max sequence length: {}", m_max_len);
  PrintInfo("  Compute precision: {}",
            m_compute_precision == Model::DataType::kFloat16 ? "FP16" : "FP32");
  PrintInfo("  SV embedding dim: {}", m_sv_dim);
}

bool EdgePipeline::ImportSpeaker(const std::string& input_path,
                                 const std::string& speaker_name) {
  PrintInfo("[EdgePipeline] Importing speaker from: {}", input_path);

  // 验证数据包
  if (!Utils::SpeakerSerializer::ValidatePackage(input_path)) {
    PrintError("[EdgePipeline] Invalid speaker package: {}", input_path);
    return false;
  }

  // 获取数据包信息
  auto package_info = Utils::SpeakerSerializer::GetPackageInfo(input_path);
  if (!package_info) {
    PrintError("[EdgePipeline] Failed to read package info: {}", input_path);
    return false;
  }

  // 确定说话人名称
  std::string final_name = speaker_name.empty() ?
      package_info->speaker_name : speaker_name;

  // 检查是否已存在
  if (m_speaker_map.find(final_name) != m_speaker_map.end()) {
    PrintWarn("[EdgePipeline] Speaker '{}' already exists, will be overwritten",
             final_name);
  }

  // 反序列化
  auto speaker_info = Utils::SpeakerSerializer::DeserializeFromFile(input_path);
  if (!speaker_info) {
    PrintError("[EdgePipeline] Failed to deserialize speaker from: {}", input_path);
    return false;
  }

  // 更新说话人名称
  speaker_info->m_speaker_name = final_name;

  // 存储到映射表
  m_speaker_map[final_name] = std::move(*speaker_info);

  PrintInfo("[EdgePipeline] Successfully imported speaker '{}', lang: {}",
            final_name, speaker_info->m_speaker_lang);

  return true;
}

std::unique_ptr<AudioTools> EdgePipeline::InferSpeaker(
    const std::string& speaker_name,
    const std::string& text,
    const std::string& text_lang,
    float temperature,
    float noise_scale,
    float speed) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter == m_speaker_map.end()) {
    PrintError("[EdgePipeline] Speaker '{}' not found", speaker_name);
    return nullptr;
  }

  const SpeakerInfo& speaker_info = iter->second;

  PrintDebug("[EdgePipeline] Inferring speaker: {}, text: {}", speaker_name, text);

  // 以下推理逻辑与 GPTSoVITSPipline::InferSpeaker 相同
  // 文本分句 - 使用 Sentence 类的流式处理方式
  std::vector<std::string> segments;
  Text::Sentence sentence(Text::Sentence::SentenceSplitMethod::Punctuation);
  
  sentence.AppendCallBack([&segments](const std::string& s) -> bool {
    segments.push_back(s);
    return true;
  });

  // 逐块添加文本（每次处理 11 个字符）
  int chunk_size = 11;
  int index = 0;
  while (index < text.size()) {
    std::string chunk = text.substr(index, chunk_size);
    sentence.Append(chunk);
    index += chunk_size;
  }
  sentence.Flush();

  if (segments.empty()) {
    PrintWarn("[EdgePipeline] No text segments to process");
    return nullptr;
  }

  std::vector<float> audio_result;

  // 遍历每个句子段落
  for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
    const std::string& segment = segments[seg_idx];

    PrintDebug("[EdgePipeline] Processing segment {}/{}: {}",
              seg_idx + 1, segments.size(), segment);

    // G2P 处理目标文本
    auto target_bert_res = m_g2p_pipline->GetPhoneAndBert(segment, text_lang);

    // 拼接参考和目标的音素
    auto ref_phones = speaker_info.m_bert_res->PhoneSeq;
    auto target_phones = target_bert_res->PhoneSeq;
    auto all_phones = ConcatTensor(ref_phones.get(), target_phones.get(), 0);

    // 拼接参考和目标的 BERT 特征
    auto ref_bert = speaker_info.m_bert_res->BertSeq;
    auto target_bert = target_bert_res->BertSeq;

    // 注意：BERT 特征需要转置以匹配模型输入格式
    // [seq_len, 1024] -> [1024, seq_len] -> [1, 1024, total_len]
    auto ref_bert_t = ref_bert->View({ref_bert->Shape()[1], ref_bert->Shape()[0]});
    auto target_bert_t = target_bert->View({target_bert->Shape()[1], target_bert->Shape()[0]});
    auto all_bert = ConcatTensor(ref_bert_t.get(), target_bert_t.get(), 1);
    auto all_bert_expanded = all_bert->View({1, all_bert->Shape()[0], all_bert->Shape()[1]});

    // 类型转换
    auto all_bert_final = all_bert_expanded->To(
        m_gpt_encoder_model->GetModel()->GetDevice(),
        m_gpt_encoder_model->GetModel()->GetInputDataType("bert_feature"));

    // 准备 prompts (VQ codes)
    auto prompts = speaker_info.m_vq_codes->View({speaker_info.m_vq_codes->Shape()[1]});
    auto prompts_final = prompts->To(
        m_gpt_encoder_model->GetModel()->GetDevice(),
        m_gpt_encoder_model->GetModel()->GetInputDataType("prompts"));

    // 准备 phoneme_ids
    auto phoneme_ids_final = all_phones->To(
        m_gpt_encoder_model->GetModel()->GetDevice(),
        m_gpt_encoder_model->GetModel()->GetInputDataType("phoneme_ids"));

    // 准备长度
    auto phoneme_ids_len = Model::Tensor::Empty(
        {1}, Model::DataType::kInt64,
        m_gpt_encoder_model->GetModel()->GetDevice());
    phoneme_ids_len->Data<int64_t>()[0] = all_phones->Shape()[0];

    // GPT Encoder 编码
    auto encoder_output = m_gpt_encoder_model->Encode(
        phoneme_ids_final.get(),
        prompts_final.get(),
        all_bert_final.get());

    // 采样第一个 token
    int64_t first_token = SampleTopK(
        encoder_output.topk_values.get(),
        encoder_output.topk_indices.get(),
        temperature);

    auto current_samples = Model::Tensor::Empty(
        {1, 1}, Model::DataType::kInt64,
        m_gpt_encoder_model->GetModel()->GetDevice());
    current_samples->Data<int64_t>()[0] = first_token;

    // 准备索引
    auto idx = Model::Tensor::Empty(
        {1}, Model::DataType::kInt64,
        m_gpt_encoder_model->GetModel()->GetDevice());
    idx->Data<int64_t>()[0] = 0;

    // 准备 x_len 和 y_len
    auto x_len = Model::Tensor::Empty(
        {1}, Model::DataType::kInt64,
        m_gpt_encoder_model->GetModel()->GetDevice());
    x_len->Data<int64_t>()[0] = ref_phones->Shape()[0];

    auto y_len = Model::Tensor::Empty(
        {1}, Model::DataType::kInt64,
        m_gpt_encoder_model->GetModel()->GetDevice());
    y_len->Data<int64_t>()[0] = target_phones->Shape()[0];

    // KV Cache
    auto k_cache = std::move(encoder_output.k_cache);
    auto v_cache = std::move(encoder_output.v_cache);

    // 语义 tokens 列表
    std::vector<std::unique_ptr<Model::Tensor>> decoded_semantic_list;
    decoded_semantic_list.push_back(speaker_info.m_vq_codes->Clone());
    decoded_semantic_list.push_back(current_samples->Clone());

    // GPT Step 自回归生成
    const int max_steps = 1500;
    const int64_t eos_token = 1024;

    for (int step = 0; step < max_steps; ++step) {
      auto step_output = m_gpt_step_model->Step(
          current_samples.get(),
          k_cache.get(),
          v_cache.get(),
          idx.get(),
          x_len.get(),
          y_len.get());

      // 采样下一个 token
      int64_t next_token = SampleTopK(
          step_output.topk_values.get(),
          step_output.topk_indices.get(),
          temperature);

      // 检查 EOS
      if (next_token == eos_token) {
        break;
      }

      // 更新 current_samples
      current_samples->Data<int64_t>()[0] = next_token;

      // 添加到列表
      decoded_semantic_list.push_back(current_samples->Clone());

      // 更新 cache
      k_cache = std::move(step_output.k_cache_new);
      v_cache = std::move(step_output.v_cache_new);

      // 更新索引
      idx->Data<int64_t>()[0]++;
    }

    // 拼接所有语义 tokens
    std::vector<Model::Tensor*> semantic_ptrs;
    for (const auto& s : decoded_semantic_list) {
      semantic_ptrs.push_back(s.get());
    }
    auto pred_semantic = Model::Tensor::Concat(semantic_ptrs, 0);

    // 去除参考部分
    auto generated_sem = pred_semantic->Slice(
        ref_phones->Shape()[0],
        pred_semantic->Shape()[0],
        0);
    generated_sem = generated_sem->View({1, 1, generated_sem->Shape()[0]});

    // SoVITS 音频生成
    auto pred_semantic_final = generated_sem->To(
        m_sovits_model->GetModel()->GetDevice(),
        m_sovits_model->GetModel()->GetInputDataType("pred_semantic"));

    auto text_seq = target_phones->To(
        m_sovits_model->GetModel()->GetDevice(),
        m_sovits_model->GetModel()->GetInputDataType("text_seq"));

    auto refer_spec = speaker_info.m_refer_spec->To(
        m_sovits_model->GetModel()->GetDevice(),
        m_sovits_model->GetModel()->GetInputDataType("refer_spec"));

    auto sv_emb = speaker_info.m_sv_emb->To(
        m_sovits_model->GetModel()->GetDevice(),
        m_sovits_model->GetModel()->GetInputDataType("sv_emb"));

    auto audio_tensor = m_sovits_model->GenerateTensor(
        pred_semantic_final.get(),
        text_seq.get(),
        refer_spec.get(),
        sv_emb.get(),
        noise_scale,
        speed);

    // 提取音频数据
    auto audio_cpu = audio_tensor->ToCPU();
    const float* audio_ptr = audio_cpu->Data<float>();
    size_t audio_size = audio_cpu->ElementCount();

    audio_result.insert(audio_result.end(), audio_ptr, audio_ptr + audio_size);
  }

  // 音频后处理：归一化
  if (!audio_result.empty()) {
    float max_amp = *std::max_element(audio_result.begin(), audio_result.end());
    if (max_amp > 0.9f) {
      float scale = 0.9f / max_amp;
      for (auto& sample : audio_result) {
        sample *= scale;
      }
    }
  }

  // 创建 AudioTools 对象
  auto result = AudioTools::FromByte(audio_result, m_sampling_rate);

  return result;
}

std::vector<std::string> EdgePipeline::ListSpeakers() const {
  std::vector<std::string> speaker_names;
  speaker_names.reserve(m_speaker_map.size());

  for (const auto& [name, _] : m_speaker_map) {
    speaker_names.push_back(name);
  }

  return speaker_names;
}

bool EdgePipeline::RemoveSpeaker(const std::string& speaker_name) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter == m_speaker_map.end()) {
    PrintError("[EdgePipeline] Speaker '{}' not found", speaker_name);
    return false;
  }

  m_speaker_map.erase(iter);
  PrintInfo("[EdgePipeline] Removed speaker: {}", speaker_name);

  return true;
}

bool EdgePipeline::HasSpeaker(const std::string& speaker_name) const {
  return m_speaker_map.find(speaker_name) != m_speaker_map.end();
}

std::string EdgePipeline::GetModelInfo() const {
  std::ostringstream oss;
  oss << "EdgePipeline Info:\n";
  oss << "  Model version: " << m_model_version << "\n";
  oss << "  Sampling rate: " << m_sampling_rate << " Hz\n";
  oss << "  Max sequence length: " << m_max_len << "\n";
  oss << "  Compute precision: "
      << (m_compute_precision == Model::DataType::kFloat16 ? "FP16" : "FP32") << "\n";
  oss << "  SV embedding dim: " << m_sv_dim << "\n";
  oss << "  Loaded speakers: " << m_speaker_map.size() << "\n";
  return oss.str();
}

// ============ Helper Methods ============

int64_t EdgePipeline::SampleTopK(const Model::Tensor* topk_values,
                                  const Model::Tensor* topk_indices,
                                  float temperature) {
  // 确保在 CPU 上
  auto values_cpu = topk_values->IsCPU() ?
      topk_values->Clone() : topk_values->ToCPU();
  auto indices_cpu = topk_indices->IsCPU() ?
      topk_indices->Clone() : topk_indices->ToCPU();

  int k = topk_values->Shape()[1];
  const float* values_ptr = values_cpu->Data<float>();
  const int64_t* indices_ptr = indices_cpu->Data<int64_t>();

  // 应用温度
  std::vector<float> probs(k);
  float max_val = *std::max_element(values_ptr, values_ptr + k);
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    probs[i] = std::exp((values_ptr[i] - max_val) / temperature);
    sum += probs[i];
  }

  // 归一化
  for (int i = 0; i < k; ++i) {
    probs[i] /= sum;
  }

  // 多项式采样
  float r = static_cast<float>(rand()) / RAND_MAX;
  float cumulative = 0.0f;
  for (int i = 0; i < k; ++i) {
    cumulative += probs[i];
    if (r <= cumulative) {
      return indices_ptr[i];
    }
  }

  return indices_ptr[k - 1];
}

std::unique_ptr<Model::Tensor> EdgePipeline::ConcatTensor(
    const Model::Tensor* a, const Model::Tensor* b, int axis) {
  std::vector<Model::Tensor*> tensors = {
      const_cast<Model::Tensor*>(a),
      const_cast<Model::Tensor*>(b)
  };
  return Model::Tensor::Concat(tensors, axis);
}

void EdgePipeline::InitializeConfig() {
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

  if (m_sovits_model) {
    m_sovits_model->SetSVDim(m_sv_dim);
  }
}

Model::DataType EdgePipeline::GetComputeDataType() const {
  return m_compute_precision;
}

}  // namespace GPTSoVITS

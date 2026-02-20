//
// Created by Huiyicc on 2026/1/17.
//
#include "GPTSoVITS/GPTSoVITSCpp.h"

#include <numeric>
#include <random>

#include "GPTSoVITS/plog.h"
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
  m_config = std::make_shared<_JsonImpl>();
  m_config->data = nlohmann::json::parse(config);
}

const SpeakerInfo& GPTSoVITSPipline::CreateSpeaker(
    const std::string& speaker_name, const std::string& ref_audio_lang,
    const std::filesystem::path& ref_audio_path,
    const std::string& ref_audio_text) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter != m_speaker_map.end()) {
    return iter->second;
  }

  PrintInfo("Creating new speaker: {}", speaker_name);

  SpeakerInfo info;
  info.m_speaker_name = speaker_name;
  info.m_speaker_lang = ref_audio_lang;

  auto refAudio = AudioTools::FromFile(ref_audio_path.string());

  auto audio16k = refAudio->ReSample(16000);
  info.m_speaker_16k = std::move(audio16k);

  // 32k (or native sr) for SoVITS
  info.m_speaker_32k = refAudio->ReSample(32000);

  // bert embeddings
  info.m_bert_res =
      m_g2p_pipline->GetPhoneAndBert(ref_audio_text, ref_audio_lang);

  // get ssl and vq codes
  info.m_ssl_content =
      m_ssl_model->GetSSLContent(info.m_speaker_16k->ReadSamples());
  info.m_vq_codes = m_vq_model->GetVQCodes(info.m_ssl_content.get());

  // reshape vq_codes from [1, 1, T] to [1, T] if needed
  if (info.m_vq_codes->Shape().size() == 3 &&
      info.m_vq_codes->Shape()[0] == 1 && info.m_vq_codes->Shape()[1] == 1) {
    info.m_vq_codes->Reshape({1, info.m_vq_codes->Shape()[2]});
  }

  auto sr = m_config->data["data"].value<int>("sampling_rate", 32000);

  info.m_refer_spec = m_spectrogram_model->ComputeSpec(
      info.m_speaker_32k->ReSample(sr)->ReadSamples());
  if (info.m_refer_spec->Shape().size() == 3) {
    info.m_refer_spec->Reshape(
        {info.m_refer_spec->Shape()[1], info.m_refer_spec->Shape()[2]});
  }

  info.m_sv_emb =
      m_sv_embedding_model->ComputeEmbedding(info.m_speaker_16k->ReadSamples());
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

  PrintInfo("Starting inference for speaker: {}", speaker_name);

  // Use Text::Sentence for multi-language text splitting
  Text::Sentence sentence(Text::Sentence::SentenceSplitMethod::Punctuation);

  std::vector<std::string> segments;
  std::vector<std::shared_ptr<Bert::BertRes>> segment_bert_res;

  // Set callback to collect segments and process them
  sentence.AppendCallBack([this, &segments, &segment_bert_res,
                           &text_lang](const std::string& seg) -> bool {
    PrintInfo(">>> [Segment Split] Processing: {}", seg);
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

  // Process text through sentence splitter
  sentence.Append(text);
  sentence.Flush();

  if (segments.empty()) {
    PrintWarn("No text segments to process");
    return nullptr;
  }

  PrintInfo("Processing {} text segments", segments.size());

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
    all_bert->Reshape({1, all_bert->Shape()[0], all_bert->Shape()[1]});
    // Prepare inputs for GPT Encoder
    auto phoneme_ids =
        Model::Tensor::Empty({1, all_phones->Shape()[0]},
                             Model::DataType::kInt64, Model::DeviceType::kCPU);
    auto phoneme_ids_len = Model::Tensor::Empty({1}, Model::DataType::kInt64,
                                                Model::DeviceType::kCPU);

    std::memcpy(phoneme_ids->Data<int64_t>(), all_phones->Data<int64_t>(),
                all_phones->ByteSize());
    phoneme_ids_len->At<int64_t>(0) = all_phones->Shape()[0];

    // Prompts from VQ codes
    auto prompts =
        Model::Tensor::Empty({1, speaker_info.m_vq_codes->Shape()[1]},
                             Model::DataType::kInt64, Model::DeviceType::kCPU);
    std::memcpy(prompts->Data<int64_t>(),
                speaker_info.m_vq_codes->Data<int64_t>(),
                speaker_info.m_vq_codes->ByteSize());

    // Run GPT Encoder
    auto encoder_output =
        m_gpt_encoder_model->Encode(phoneme_ids.get(), phoneme_ids_len.get(),
                                    prompts.get(), all_bert.get());

    // Sample first token
    int64_t first_token =
        SampleTopK(encoder_output.topk_values.get(),
                   encoder_output.topk_indices.get(), temperature);

    // Prepare semantic list
    std::vector<std::unique_ptr<Model::Tensor>> decoded_semantic_list;
    decoded_semantic_list.push_back(speaker_info.m_vq_codes->Clone());

    auto first_token_tensor = Model::Tensor::Empty(
        {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
    first_token_tensor->At<int64_t>(0) = first_token;
    decoded_semantic_list.push_back(std::move(first_token_tensor));

    // GPT Step loop
    auto current_samples = first_token_tensor->Clone();
    auto k_cache = std::move(encoder_output.k_cache);
    auto v_cache = std::move(encoder_output.v_cache);
    auto x_len = std::move(encoder_output.x_len);
    auto y_len = std::move(encoder_output.y_len);

    int max_steps = 1500;
    int64_t eos_token = 1024;
    int steps = 0;

    for (int i = 0; i < max_steps; ++i) {
      auto idx_tensor = Model::Tensor::Empty({1}, Model::DataType::kInt64,
                                             Model::DeviceType::kCPU);
      idx_tensor->At<int64_t>(0) = i;

      auto step_output = m_gpt_step_model->Step(
          current_samples.get(), k_cache.get(), v_cache.get(), idx_tensor.get(),
          x_len.get(), y_len.get());

      // Update cache
      k_cache = std::move(step_output.k_cache_new);
      v_cache = std::move(step_output.v_cache_new);

      // Sample next token
      int64_t next_token =
          SampleTopK(step_output.topk_values.get(),
                     step_output.topk_indices.get(), temperature);

      auto next_token_tensor = Model::Tensor::Empty(
          {1, 1}, Model::DataType::kInt64, Model::DeviceType::kCPU);
      next_token_tensor->At<int64_t>(0) = next_token;
      decoded_semantic_list.push_back(std::move(next_token_tensor));
      current_samples = next_token_tensor->Clone();

      steps++;

      // Check for EOS
      if (next_token == eos_token) {
        PrintInfo("Generated {} tokens before EOS", steps);
        break;
      }
    }

    // Concatenate all semantic tokens
    std::vector<Model::Tensor*> semantic_ptrs;
    for (const auto& s : decoded_semantic_list) {
      semantic_ptrs.push_back(s.get());
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


    break;
  }

  return AudioTools::FromEmpty(32000);
}

namespace {

// Helper function to sample from top-k
int64_t sample_top_k(const Model::Tensor* topk_values,
                     const Model::Tensor* topk_indices, float temperature) {
  auto values = topk_values->Data<float>();
  auto indices = topk_indices->Data<int64_t>();
  int k = topk_values->ElementCount();

  // Apply temperature
  std::vector<float> probs(values, values + k);
  if (temperature != 1.0f) {
    for (auto& p : probs) {
      p /= temperature;
    }
  }

  // Find max for numerical stability
  float max_val = *std::max_element(probs.begin(), probs.end());
  for (auto& p : probs) {
    p -= max_val;
  }

  // Apply softmax
  for (auto& p : probs) {
    p = std::exp(p);
  }
  float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
  for (auto& p : probs) {
    p /= sum;
  }

  // Sample
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  int choice = dist(gen);

  return indices[choice];
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
}  // namespace GPTSoVITS
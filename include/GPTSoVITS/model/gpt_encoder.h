//
// Created by Huiyicc on 2026/2/17.
//

#ifndef GSV_CPP_GPT_ENCODER_H
#define GSV_CPP_GPT_ENCODER_H

#include <memory>
#include <vector>

#include "GPTSoVITS/model/base.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS::Model {

/**
 * @brief GPT Encoder output structure
 */
struct GPTEncoderOutput {
  std::unique_ptr<Tensor> topk_values;      // [1, top_k]
  std::unique_ptr<Tensor> topk_indices;     // [1, top_k]
  std::unique_ptr<Tensor> k_cache;          // [num_layers, batch, num_heads, seq_len, head_dim]
  std::unique_ptr<Tensor> v_cache;          // [num_layers, batch, num_heads, seq_len, head_dim]
  std::unique_ptr<Tensor> x_len;            // [1] - prompt length
  std::unique_ptr<Tensor> y_len;            // [1] - target text length
};

/**
 * @brief GPT Encoder Model
 *
 * GPT Encoder processes the phoneme sequence and BERT features to generate
 * the initial semantic tokens and KV cache for the GPT decoder.
 */
class GPTEncoderModel {
protected:
  std::unique_ptr<BaseModel> m_model;
  int m_num_layers = 0;
  int m_num_heads = 0;
  int m_head_dim = 0;
  int m_max_seq_len = 1500;

public:
  GPTEncoderModel() = default;
  virtual ~GPTEncoderModel() = default;

  /**
   * @brief Initialize model with backend
   * @tparam MODEL_BACKEND Backend type (e.g., ONNXBackend)
   * @param model_path Path to gpt_encoder.onnx
   * @param device Device to run on
   * @param work_thread_num Number of threads
   */
  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path,
            const Device& device = DeviceType::kCPU,
            int work_thread_num = 1) {
    m_model = std::make_unique<MODEL_BACKEND>();
    m_model->Load(model_path, device, work_thread_num);

    // Parse cache shape from output metadata
    auto output_names = m_model->GetOutputNames();
    for (const auto& name : output_names) {
      if (name == "k_cache" || name == "v_cache") {
        // Cache shape: [num_layers, batch, num_heads, seq_len, head_dim]
        auto dtype = m_model->GetOutputDataType(name);
        // Will be updated after first inference based on actual shape
      }
    }
  }

  /**
   * @brief Run GPT Encoder inference
   *
   * @param phoneme_ids Concatenated phoneme IDs [1, seq_len]
   * @param phoneme_ids_len Length of phoneme sequence [1]
   * @param prompts Reference semantic tokens from VQ [1, prompt_len]
   * @param bert_feature BERT features [1, 1024, seq_len]
   * @return GPTEncoderOutput containing topk, indices, and KV cache
   */
  GPTEncoderOutput Encode(
      Tensor* phoneme_ids,
      Tensor* phoneme_ids_len,
      Tensor* prompts,
      Tensor* bert_feature);

  /**
   * @brief Get the number of transformer layers
   */
  [[nodiscard]] int GetNumLayers() const { return m_num_layers; }

  /**
   * @brief Get the number of attention heads
   */
  [[nodiscard]] int GetNumHeads() const { return m_num_heads; }

  /**
   * @brief Get the head dimension
   */
  [[nodiscard]] int GetHeadDim() const { return m_head_dim; }

  /**
   * @brief Get max sequence length
   */
  [[nodiscard]] int GetMaxSeqLen() const { return m_max_seq_len; }

  /**
   * @brief Get the underlying model
   */
  [[nodiscard]] const BaseModel* GetModel() const { return m_model.get(); }
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_GPT_ENCODER_H
//
// Created by Huiyicc on 2026/2/17.
//

#ifndef GSV_CPP_GPT_STEP_H
#define GSV_CPP_GPT_STEP_H

#include <memory>
#include <vector>

#include "GPTSoVITS/model/base.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS::Model {

/**
 * @brief GPT Step output structure
 */
struct GPTStepOutput {
  std::unique_ptr<Tensor> topk_values;      // [1, top_k]
  std::unique_ptr<Tensor> topk_indices;     // [1, top_k]
  std::unique_ptr<Tensor> k_cache_new;      // [num_layers, batch, num_heads, seq_len, head_dim]
  std::unique_ptr<Tensor> v_cache_new;      // [num_layers, batch, num_heads, seq_len, head_dim]
};

/**
 * @brief GPT Step Model
 *
 * GPT Step performs single-step autoregressive generation.
 * This implementation supports zero-copy IO binding to minimize memory transfers
 * between CPU and GPU, which is critical for performance during multi-step generation.
 */
class GPTStepModel {
protected:
  std::unique_ptr<BaseModel> m_model;
  int m_top_k = 5;
  bool m_supports_iobinding = false;
  DataType m_cache_dtype = DataType::kFloat32;

public:
  GPTStepModel() = default;
  virtual ~GPTStepModel() = default;

  /**
   * @brief Initialize model with backend
   * @tparam MODEL_BACKEND Backend type (e.g., ONNXBackend)
   * @param model_path Path to gpt_step.onnx
   * @param device Device to run on
   * @param work_thread_num Number of threads
   */
  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path,
            const Device& device = DeviceType::kCPU,
            int work_thread_num = 1) {
    m_model = std::make_unique<MODEL_BACKEND>();
    m_model->Load(model_path, device, work_thread_num);

    // Detect cache data type
    auto input_names = m_model->GetInputNames();
    for (const auto& name : input_names) {
      if (name == "k_cache" || name == "v_cache") {
        m_cache_dtype = m_model->GetInputDataType(name);
      }
    }

    // Check if backend supports IO binding
    // This will be set based on runtime detection in implementation
  }

  /**
   * @brief Run single GPT step inference (standard forward)
   *
   * @param samples Previous token sample [1, 1]
   * @param k_cache Key cache [num_layers, batch, num_heads, seq_len, head_dim]
   * @param v_cache Value cache [num_layers, batch, num_heads, seq_len, head_dim]
   * @param idx Current step index [1]
   * @param x_len Prompt length [1]
   * @param y_len Target length [1]
   * @return GPTStepOutput containing next topk, indices, and updated cache
   */
  GPTStepOutput Step(Tensor* samples,
                     Tensor* k_cache,
                     Tensor* v_cache,
                     Tensor* idx,
                     Tensor* x_len,
                     Tensor* y_len);

  /**
   * @brief Run GPT step with IO binding for zero-copy (if supported)
   *
   * This method uses pre-allocated tensors to avoid memory allocations
   * during generation loops. It's more efficient for multi-step generation.
   *
   * @param samples Previous token sample [1, 1]
   * @param k_cache_in Input key cache (will be consumed)
   * @param v_cache_in Input value cache (will be consumed)
   * @param k_cache_out Output key cache (will be written to)
   * @param v_cache_out Output value cache (will be written to)
   * @param idx Current step index [1]
   * @param x_len Prompt length [1]
   * @param y_len Target length [1]
   * @param topk_values_out Output buffer for topk values [1, top_k]
   * @param topk_indices_out Output buffer for topk indices [1, top_k]
   * @return true if successful
   */
  bool StepWithIOBinding(Tensor* samples,
                         Tensor* k_cache_in,
                         Tensor* v_cache_in,
                         Tensor* k_cache_out,
                         Tensor* v_cache_out,
                         Tensor* idx,
                         Tensor* x_len,
                         Tensor* y_len,
                         Tensor* topk_values_out,
                         Tensor* topk_indices_out);

  /**
   * @brief Check if IO binding is supported
   */
  [[nodiscard]] bool SupportsIOBinding() const { return m_supports_iobinding; }

  /**
   * @brief Get cache data type
   */
  [[nodiscard]] DataType GetCacheDType() const { return m_cache_dtype; }

  /**
   * @brief Get the underlying model
   */
  [[nodiscard]] const BaseModel* GetModel() const { return m_model.get(); }
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_GPT_STEP_H
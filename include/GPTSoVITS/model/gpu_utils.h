//
// Created by iFlow CLI on 2026/2/20.
//

#ifndef GPT_SOVITS_CPP_GPU_UTILS_H
#define GPT_SOVITS_CPP_GPU_UTILS_H

#include <memory>

#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS::Model::GPU {

/**
 * @brief GPU 采样工具类
 */
class GPUSampler {
public:
  GPUSampler();
  ~GPUSampler();

  /**
   * @brief 在 GPU 上进行 top-k 采样
   * @param topk_values Top-K 值张量 [batch, k]
   * @param topk_indices Top-K 索引张量 [batch, k]
   * @param temperature 温度参数
   * @return 采样的索引 [batch, 1]
   */
  std::unique_ptr<Tensor> SampleTopK(
      const Tensor* topk_values,
      const Tensor* topk_indices,
      float temperature = 1.0f);

  /**
   * @brief 在 GPU 上进行多项式采样（从完整概率分布）
   * @param logits Logits 张量 [batch, vocab_size]
   * @param temperature 温度参数
   * @return 采样的索引 [batch, 1]
   */
  std::unique_ptr<Tensor> SampleMultinomial(
      const Tensor* logits,
      float temperature = 1.0f);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * @brief GPU 类型转换工具（零拷贝优化）
 *
 * 该类提供在 GPU 上直接进行类型转换的功能，
 * 避免 D2H + CPU转换 + H2D 的往返操作。
 */
class GPUTypeConverter {
public:
  /**
   * @brief 在 GPU 上转换 Tensor 数据类型
   * @param src 源 Tensor
   * @param target_dtype 目标数据类型
   * @return 转换后的 Tensor
   */
  static std::unique_ptr<Tensor> ConvertType(
      const Tensor* src,
      DataType target_dtype);

  /**
   * @brief 检查是否支持该转换
   * @param src_dtype 源数据类型
   * @param target_dtype 目标数据类型
   * @return 是否支持
   */
  static bool IsConversionSupported(DataType src_dtype, DataType target_dtype);
};

/**
 * @brief GPU Cache 优化工具
 *
 * 该类提供 KV Cache 的零拷贝管理功能。
 */
class GPUCacheManager {
public:
  /**
   * @brief 预分配 Cache 缓冲区
   * @param max_seq_len 最大序列长度
   * @param num_layers 层数
   * @param num_heads 头数
   * @param head_dim 头维度
   * @param dtype 数据类型
   * @param device 设备
   */
  static std::unique_ptr<Tensor> PreallocateKVCache(
      int64_t max_seq_len,
      int num_layers,
      int num_heads,
      int head_dim,
      DataType dtype,
      const Device& device);

  /**
   * @brief 在 GPU 上更新 Cache（零拷贝）
   * @param cache 目标 Cache 张量
   * @param new_data 新数据张量
   * @param offset 写入偏移
   */
  static void UpdateCacheInPlace(
      Tensor* cache,
      const Tensor* new_data,
      int64_t offset);
};

}  // namespace GPTSoVITS::Model::GPU

#endif  // GPT_SOVITS_CPP_GPU_UTILS_H
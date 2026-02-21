//
// Created by iFlow CLI on 2026/2/20.
//

#ifndef GPT_SOVITS_CPP_SPEAKER_SERIALIZER_H
#define GPT_SOVITS_CPP_SPEAKER_SERIALIZER_H

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

#include "GPTSoVITS/GPTSoVITSCpp.h"
#include "GPTSoVITS/model/tensor.h"
#include "GPTSoVITS/Utils/exception.h"

namespace GPTSoVITS::Utils {

/**
 * @brief 说话人数据包格式
 *
 * 文件结构:
 * [JSON Header (UTF-8)][Binary Data]
 *
 * JSON Header 包含:
 * - version: 数据包版本
 * - speaker_name: 说话人名称
 * - speaker_lang: 说话人语言
 * - metadata: 元数据（创建时间、模型版本等）
 * - tensors: 各张量的元数据（形状、类型、偏移、大小）
 *
 * Binary Data 包含:
 * - PhoneSeq 数据
 * - BertSeq 数据
 * - VQ Codes 数据
 * - Refer Spec 数据
 * - SV Embedding 数据
 */
class SpeakerSerializer {
public:
  // 数据包版本
  static constexpr uint32_t CURRENT_VERSION = 1;

  /**
   * @brief 序列化 Tensor 元数据
   */
  struct TensorMetadata {
    std::string name;
    std::vector<int64_t> shape;
    Model::DataType dtype;
    uint64_t offset;  // 在 binary data 中的偏移（字节）
    uint64_t size;    // 数据大小（字节）

    nlohmann::json ToJson() const;
    static TensorMetadata FromJson(const nlohmann::json& j);
  };

  /**
   * @brief 数据包头信息
   */
  struct PackageHeader {
    uint32_t version;
    std::string speaker_name;
    std::string speaker_lang;

    // 元数据
    struct Metadata {
      std::string created_at;        // ISO 8601 格式
      std::string model_version;     // "v2" or "v2ProPlus"
      int sv_dim;                    // SV embedding 维度
      int max_seq_len;               // 最大序列长度
    } metadata;

    // 张量元数据
    std::unordered_map<std::string, TensorMetadata> tensors;

    nlohmann::json ToJson() const;
    static PackageHeader FromJson(const nlohmann::json& j);
  };

  /**
   * @brief 序列化 SpeakerInfo 到文件
   * @param speaker_info 说话人信息
   * @param output_path 输出文件路径
   * @param include_audio 是否包含音频数据（可选，增加文件大小）
   * @return 是否成功
   */
  static bool SerializeToFile(
      const SpeakerInfo& speaker_info,
      const std::string& output_path,
      bool include_audio = false);

  /**
   * @brief 从文件反序列化 SpeakerInfo
   * @param input_path 输入文件路径
   * @return SpeakerInfo 对象（如果失败则返回 nullptr）
   */
  static std::unique_ptr<SpeakerInfo> DeserializeFromFile(
      const std::string& input_path);

  /**
   * @brief 序列化到内存缓冲区
   * @param speaker_info 说话人信息
   * @param output 输出缓冲区
   * @param include_audio 是否包含音频数据
   * @return 是否成功
   */
  static bool SerializeToBuffer(
      const SpeakerInfo& speaker_info,
      std::vector<uint8_t>& output,
      bool include_audio = false);

  /**
   * @brief 从内存缓冲区反序列化
   * @param buffer 输入缓冲区
   * @return SpeakerInfo 对象（如果失败则返回 nullptr）
   */
  static std::unique_ptr<SpeakerInfo> DeserializeFromBuffer(
      const std::vector<uint8_t>& buffer);

  /**
   * @brief 验证数据包文件完整性
   * @param file_path 文件路径
   * @return 是否有效
   */
  static bool ValidatePackage(const std::string& file_path);

  /**
   * @brief 获取数据包信息（不加载完整数据）
   * @param file_path 文件路径
   * @return PackageHeader（如果失败则返回 nullptr）
   */
  static std::unique_ptr<PackageHeader> GetPackageInfo(
      const std::string& file_path);

  /**
   * @brief 获取数据包大小（不加载完整数据）
   * @param file_path 文件路径
   * @return 文件大小（字节）
   */
  static uint64_t GetPackageSize(const std::string& file_path);

private:
  /**
   * @brief 序列化 Tensor 到二进制数据
   * @param tensor Tensor 对象
   * @param buffer 输出缓冲区
   * @param offset 起始偏移
   * @return TensorMetadata
   */
  static TensorMetadata SerializeTensor(
      const Model::Tensor* tensor,
      std::vector<uint8_t>& buffer,
      uint64_t offset);

  /**
   * @brief 从二进制数据反序列化 Tensor
   * @param metadata Tensor 元数据
   * @param buffer 输入缓冲区
   * @return Tensor 对象
   */
  static std::unique_ptr<Model::Tensor> DeserializeTensor(
      const TensorMetadata& metadata,
      const std::vector<uint8_t>& buffer);

  /**
   * @brief 将 Tensor 数据扁平化到 CPU 缓冲区
   * @param tensor Tensor 对象
   * @return 字节缓冲区
   */
  static std::vector<uint8_t> FlattenTensor(const Model::Tensor* tensor);

  /**
   * @brief 从 CPU 缓冲区创建 Tensor
   * @param buffer 字节缓冲区
   * @param metadata Tensor 元数据
   * @return Tensor 对象
   */
  static std::unique_ptr<Model::Tensor> CreateTensorFromBuffer(
      const std::vector<uint8_t>& buffer,
      const TensorMetadata& metadata);

  /**
   * @brief 获取当前时间（ISO 8601 格式）
   * @return 时间字符串
   */
  static std::string GetCurrentTime();

  /**
   * @brief 检查版本兼容性
   * @param version 数据包版本
   * @return 是否兼容
   */
  static bool IsVersionCompatible(uint32_t version);
};

}  // namespace GPTSoVITS::Utils

#endif  // GPT_SOVITS_CPP_SPEAKER_SERIALIZER_H
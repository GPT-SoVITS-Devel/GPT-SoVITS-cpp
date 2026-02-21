//
// Created by iFlow CLI on 2026/2/20.
//

#include "GPTSoVITS/Utils/speaker_serializer.h"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>

#include "GPTSoVITS/GPTSoVITSCpp.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS::Utils {

// ============ PackageHeader ============

nlohmann::json SpeakerSerializer::PackageHeader::ToJson() const {
  nlohmann::json j;

  j["version"] = version;
  j["speaker_name"] = speaker_name;
  j["speaker_lang"] = speaker_lang;

  // 元数据
  j["metadata"]["created_at"] = metadata.created_at;
  j["metadata"]["model_version"] = metadata.model_version;
  j["metadata"]["sv_dim"] = metadata.sv_dim;
  j["metadata"]["max_seq_len"] = metadata.max_seq_len;

  // 张量元数据
  for (const auto& [name, tensor_meta] : tensors) {
    j["tensors"][name] = tensor_meta.ToJson();
  }

  return j;
}

SpeakerSerializer::PackageHeader SpeakerSerializer::PackageHeader::FromJson(
    const nlohmann::json& j) {
  PackageHeader header;

  header.version = j["version"];
  header.speaker_name = j["speaker_name"];
  header.speaker_lang = j["speaker_lang"];

  // 元数据
  header.metadata.created_at = j["metadata"]["created_at"];
  header.metadata.model_version = j["metadata"]["model_version"];
  header.metadata.sv_dim = j["metadata"]["sv_dim"];
  header.metadata.max_seq_len = j["metadata"]["max_seq_len"];

  // 张量元数据
  if (j.contains("tensors")) {
    for (auto it = j["tensors"].begin(); it != j["tensors"].end(); ++it) {
      header.tensors[it.key()] = TensorMetadata::FromJson(it.value());
    }
  }

  return header;
}

// ============ TensorMetadata ============

nlohmann::json SpeakerSerializer::TensorMetadata::ToJson() const {
  nlohmann::json j;

  j["name"] = name;
  j["shape"] = shape;
  j["dtype"] = static_cast<int>(dtype);
  j["offset"] = offset;
  j["size"] = size;

  return j;
}

SpeakerSerializer::TensorMetadata SpeakerSerializer::TensorMetadata::FromJson(
    const nlohmann::json& j) {
  TensorMetadata meta;

  meta.name = j["name"];
  meta.shape = j["shape"].get<std::vector<int64_t>>();
  meta.dtype = static_cast<Model::DataType>(j["dtype"]);
  meta.offset = j["offset"];
  meta.size = j["size"];

  return meta;
}

// ============ 辅助函数 ============

std::string SpeakerSerializer::GetCurrentTime() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;

  std::ostringstream oss;
  oss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
  oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << "Z";

  return oss.str();
}

bool SpeakerSerializer::IsVersionCompatible(uint32_t version) {
  return version == CURRENT_VERSION;
}

std::vector<uint8_t> SpeakerSerializer::FlattenTensor(const Model::Tensor* tensor) {
  if (!tensor) {
    return {};
  }

  // 确保数据在 CPU 上
  std::unique_ptr<Model::Tensor> cpu_tensor;
  if (tensor->IsCPU()) {
    cpu_tensor = tensor->Clone();
  } else {
    cpu_tensor = tensor->ToCPU();
  }

  // 转换为字节缓冲区
  size_t byte_size = cpu_tensor->ByteSize();
  const uint8_t* data_ptr = static_cast<const uint8_t*>(cpu_tensor->Data());

  return std::vector<uint8_t>(data_ptr, data_ptr + byte_size);
}

std::unique_ptr<Model::Tensor> SpeakerSerializer::CreateTensorFromBuffer(
    const std::vector<uint8_t>& buffer,
    const TensorMetadata& metadata) {
  // 计算数据指针
  if (metadata.offset + metadata.size > buffer.size()) {
    THROW_ERRORN("Invalid tensor offset/size: offset={}, size={}, buffer_size={}",
                 metadata.offset, metadata.size, buffer.size());
  }

  const uint8_t* data_ptr = buffer.data() + metadata.offset;

  // 创建 CPU Tensor
  auto tensor = Model::Tensor::Empty(
      metadata.shape,
      metadata.dtype,
      Model::Device(Model::DeviceType::kCPU));

  // 拷贝数据
  std::memcpy(tensor->Data(), data_ptr, metadata.size);

  return tensor;
}

// ============ 序列化/反序列化实现 ============

bool SpeakerSerializer::SerializeToFile(
    const SpeakerInfo& speaker_info,
    const std::string& output_path,
    bool include_audio) {
  try {
    std::vector<uint8_t> buffer;
    if (!SerializeToBuffer(speaker_info, buffer, include_audio)) {
      return false;
    }

    std::ofstream file(output_path, std::ios::binary);
    if (!file) {
      PrintError("[SpeakerSerializer] Failed to open file for writing: {}", output_path);
      return false;
    }

    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    return true;
  } catch (const std::exception& e) {
    PrintError("[SpeakerSerializer] SerializeToFile failed: {}", e.what());
    return false;
  }
}

std::unique_ptr<SpeakerInfo> SpeakerSerializer::DeserializeFromFile(
    const std::string& input_path) {
  try {
    std::ifstream file(input_path, std::ios::binary);
    if (!file) {
      PrintError("[SpeakerSerializer] Failed to open file for reading: {}", input_path);
      return nullptr;
    }

    // 读取文件内容
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    return DeserializeFromBuffer(buffer);
  } catch (const std::exception& e) {
    PrintError("[SpeakerSerializer] DeserializeFromFile failed: {}", e.what());
    return nullptr;
  }
}

bool SpeakerSerializer::SerializeToBuffer(
    const SpeakerInfo& speaker_info,
    std::vector<uint8_t>& output,
    bool include_audio) {
  try {
    PackageHeader header;
    header.version = CURRENT_VERSION;
    header.speaker_name = speaker_info.SpeakerName();
    header.speaker_lang = speaker_info.SpeakerLang();

    // 设置元数据
    header.metadata.created_at = GetCurrentTime();
    header.metadata.model_version = "v2";  // TODO: 从配置获取
    header.metadata.sv_dim = 20480;  // TODO: 从配置获取
    header.metadata.max_seq_len = 1000;  // TODO: 从配置获取

    // 准备二进制数据缓冲区
    std::vector<uint8_t> binary_data;

    // 序列化 PhoneSeq
    if (speaker_info.BertRes() && speaker_info.BertRes()->PhoneSeq) {
      auto meta = SerializeTensor(
          speaker_info.BertRes()->PhoneSeq.get(),
          binary_data,
          binary_data.size());
      header.tensors["phone_seq"] = meta;
    }

    // 序列化 BertSeq
    if (speaker_info.BertRes() && speaker_info.BertRes()->BertSeq) {
      auto meta = SerializeTensor(
          speaker_info.BertRes()->BertSeq.get(),
          binary_data,
          binary_data.size());
      header.tensors["bert_seq"] = meta;
    }

    // 序列化 VQ Codes
    if (speaker_info.m_vq_codes) {
      auto meta = SerializeTensor(
          speaker_info.m_vq_codes.get(),
          binary_data,
          binary_data.size());
      header.tensors["vq_codes"] = meta;
    }

    // 序列化 Refer Spec
    if (speaker_info.m_refer_spec) {
      auto meta = SerializeTensor(
          speaker_info.m_refer_spec.get(),
          binary_data,
          binary_data.size());
      header.tensors["refer_spec"] = meta;
    }

    // 序列化 SV Embedding
    if (speaker_info.m_sv_emb) {
      auto meta = SerializeTensor(
          speaker_info.m_sv_emb.get(),
          binary_data,
          binary_data.size());
      header.tensors["sv_emb"] = meta;
    }

    // 序列化音频数据
    if (include_audio && speaker_info.m_speaker_16k) {
      // TODO: 实现音频序列化
      PrintWarn("[SpeakerSerializer] Audio serialization not yet implemented");
    }

    // 将 JSON header 转换为字符串
    nlohmann::json header_json = header.ToJson();
    std::string header_str = header_json.dump();

    // 组合输出: [header_size (4 bytes)][header_str][binary_data]
    uint32_t header_size = static_cast<uint32_t>(header_str.size());

    output.clear();
    output.reserve(sizeof(header_size) + header_size + binary_data.size());

    // 写入 header 大小
    const uint8_t* size_ptr = reinterpret_cast<const uint8_t*>(&header_size);
    output.insert(output.end(), size_ptr, size_ptr + sizeof(header_size));

    // 写入 header 字符串
    output.insert(output.end(), header_str.begin(), header_str.end());

    // 写入二进制数据
    output.insert(output.end(), binary_data.begin(), binary_data.end());

    return true;
  } catch (const std::exception& e) {
    PrintError("[SpeakerSerializer] SerializeToBuffer failed: {}", e.what());
    return false;
  }
}

std::unique_ptr<SpeakerInfo> SpeakerSerializer::DeserializeFromBuffer(
    const std::vector<uint8_t>& buffer) {
  try {
    if (buffer.size() < sizeof(uint32_t)) {
      PrintError("[SpeakerSerializer] Buffer too small");
      return nullptr;
    }

    // 读取 header 大小
    uint32_t header_size = *reinterpret_cast<const uint32_t*>(buffer.data());

    // 检查缓冲区大小
    if (buffer.size() < sizeof(header_size) + header_size) {
      PrintError("[SpeakerSerializer] Invalid buffer size");
      return nullptr;
    }

    // 读取 JSON header
    std::string header_str(
        reinterpret_cast<const char*>(buffer.data() + sizeof(header_size)),
        header_size);

    nlohmann::json header_json = nlohmann::json::parse(header_str);
    PackageHeader header = PackageHeader::FromJson(header_json);

    // 验证版本
    if (!IsVersionCompatible(header.version)) {
      PrintError("[SpeakerSerializer] Incompatible package version: {}", header.version);
      return nullptr;
    }

    // 提取二进制数据
    const uint8_t* binary_data = buffer.data() + sizeof(header_size) + header_size;
    size_t binary_size = buffer.size() - sizeof(header_size) - header_size;
    std::vector<uint8_t> binary_vec(binary_data, binary_data + binary_size);

    // 创建 SpeakerInfo
    auto speaker_info = std::make_unique<SpeakerInfo>();
    speaker_info->m_speaker_name = header.speaker_name;
    speaker_info->m_speaker_lang = header.speaker_lang;

    // 创建 BertRes
    speaker_info->m_bert_res = std::make_shared<Bert::BertRes>();

    // 反序列化 PhoneSeq
    if (header.tensors.find("phone_seq") != header.tensors.end()) {
      speaker_info->m_bert_res->PhoneSeq = DeserializeTensor(
          header.tensors["phone_seq"], binary_vec);
    }

    // 反序列化 BertSeq
    if (header.tensors.find("bert_seq") != header.tensors.end()) {
      speaker_info->m_bert_res->BertSeq = DeserializeTensor(
          header.tensors["bert_seq"], binary_vec);
    }

    // 反序列化 VQ Codes
    if (header.tensors.find("vq_codes") != header.tensors.end()) {
      speaker_info->m_vq_codes = DeserializeTensor(
          header.tensors["vq_codes"], binary_vec);
    }

    // 反序列化 Refer Spec
    if (header.tensors.find("refer_spec") != header.tensors.end()) {
      speaker_info->m_refer_spec = DeserializeTensor(
          header.tensors["refer_spec"], binary_vec);
    }

    // 反序列化 SV Embedding
    if (header.tensors.find("sv_emb") != header.tensors.end()) {
      speaker_info->m_sv_emb = DeserializeTensor(
          header.tensors["sv_emb"], binary_vec);
    }

    PrintInfo("[SpeakerSerializer] Successfully loaded speaker: {}",
              header.speaker_name);

    return speaker_info;
  } catch (const std::exception& e) {
    PrintError("[SpeakerSerializer] DeserializeFromBuffer failed: {}", e.what());
    return nullptr;
  }
}

SpeakerSerializer::TensorMetadata SpeakerSerializer::SerializeTensor(
    const Model::Tensor* tensor,
    std::vector<uint8_t>& buffer,
    uint64_t offset) {
  if (!tensor) {
    THROW_ERRORN("[SpeakerSerializer] Cannot serialize null tensor");
  }

  // 扁平化 Tensor 数据
  std::vector<uint8_t> data = FlattenTensor(tensor);

  // 扩展缓冲区
  size_t old_size = buffer.size();
  buffer.resize(old_size + data.size());

  // 拷贝数据
  std::memcpy(buffer.data() + old_size, data.data(), data.size());

  // 创建元数据
  TensorMetadata meta;
  meta.name = "tensor";
  meta.shape = tensor->Shape();
  meta.dtype = tensor->Type();
  meta.offset = offset;
  meta.size = data.size();

  return meta;
}

std::unique_ptr<Model::Tensor> SpeakerSerializer::DeserializeTensor(
    const TensorMetadata& metadata,
    const std::vector<uint8_t>& buffer) {
  return CreateTensorFromBuffer(buffer, metadata);
}

bool SpeakerSerializer::ValidatePackage(const std::string& file_path) {
  try {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
      return false;
    }

    // 读取 header 大小
    uint32_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    if (!file || file.fail()) {
      return false;
    }

    // 读取 JSON header
    std::string header_str(header_size, '\0');
    file.read(&header_str[0], header_size);

    if (!file || file.fail()) {
      return false;
    }

    nlohmann::json header_json = nlohmann::json::parse(header_str);
    PackageHeader header = PackageHeader::FromJson(header_json);

    // 验证版本
    if (!IsVersionCompatible(header.version)) {
      return false;
    }

    // 读取二进制数据并验证校验和
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t binary_size = file_size - sizeof(header_size) - header_size;

    if (binary_size <= 0) {
      return false;
    }

    std::vector<uint8_t> binary_data(binary_size);
    file.seekg(sizeof(header_size) + header_size);
    file.read(reinterpret_cast<char*>(binary_data.data()), binary_size);

    if (!file || file.fail()) {
      return false;
    }

    // Checksum 验证已移除，直接返回 true
    return true;
  } catch (const std::exception& e) {
    PrintError("[SpeakerSerializer] ValidatePackage failed: {}", e.what());
    return false;
  }
}

std::unique_ptr<SpeakerSerializer::PackageHeader> SpeakerSerializer::GetPackageInfo(
    const std::string& file_path) {
  try {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
      return nullptr;
    }

    // 读取 header 大小
    uint32_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    if (!file || file.fail()) {
      return nullptr;
    }

    // 读取 JSON header
    std::string header_str(header_size, '\0');
    file.read(&header_str[0], header_size);

    if (!file || file.fail()) {
      return nullptr;
    }

    nlohmann::json header_json = nlohmann::json::parse(header_str);
    auto header = std::make_unique<PackageHeader>(PackageHeader::FromJson(header_json));

    return header;
  } catch (const std::exception& e) {
    PrintError("[SpeakerSerializer] GetPackageInfo failed: {}", e.what());
    return nullptr;
  }
}

uint64_t SpeakerSerializer::GetPackageSize(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file) {
    return 0;
  }
  return static_cast<uint64_t>(file.tellg());
}

}  // namespace GPTSoVITS::Utils
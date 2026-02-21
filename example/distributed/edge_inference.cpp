//
// 边缘：加载说话人数据包并推理
//
// 场景：边缘设备（如嵌入式系统、移动设备）加载预打包的说话人数据
// 优势：仅需4个推理模型，大幅降低内存占用
//

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "GPTSoVITS/EdgePipeline.h"
#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/Utils/speaker_serializer.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"

#ifdef _HOST_WINDOWS_
auto device = GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCUDA, 0);
#else
auto device = GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCPU, 0);
#endif

#ifdef _HOST_WINDOWS_
#define MODEL_PATH R"(F:\Engcode\AIAssistant\GPT-SoVITS-Devel\GPT-SoVITS_minimal_inference\onnx_export\firefly_v2_proplus_fp16)"
#else
#define MODEL_PATH R"(/Users/huiyi/code/python/GPT-SoVITS_minimal_inference/onnx_export/firefly_v2_proplus_fp16)"
#endif

std::string readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("无法打开文件");
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
  std::system("chcp 65001");
#endif

  try {
    std::cout << "========================================" << std::endl;
    std::cout << "  加载说话人数据包" << std::endl;
    std::cout << "========================================" << std::endl;

    // 解析命令行参数
    std::string speaker_package = "firefly.gsppkg";
    std::string text = "你好，这是一段测试文本，演示边缘设备推理能力。";
    std::string text_lang = "zh";
    std::string speaker_name = "firefly";
    std::string output_path = "edge_output.wav";

    if (argc >= 2) speaker_package = argv[1];
    if (argc >= 3) text = argv[2];
    if (argc >= 4) text_lang = argv[3];
    if (argc >= 5) speaker_name = argv[4];
    if (argc >= 6) output_path = argv[5];

    std::filesystem::path modelPath = FS_PATH(MODEL_PATH);

    std::cout << "\n配置信息:" << std::endl;
    std::cout << "  模型路径: " << MODEL_PATH << std::endl;
    std::cout << "  说话人数据包: " << speaker_package << std::endl;
    std::cout << "  文本: " << text << std::endl;
    std::cout << "  语言: " << text_lang << std::endl;
    std::cout << "  设备: " << (device.type == GPTSoVITS::Model::DeviceType::kCUDA ? "CUDA" : "CPU") << std::endl;

    // 验证数据包
    std::cout << "\n[1/5] 验证说话人数据包..." << std::endl;
    if (!GPTSoVITS::Utils::SpeakerSerializer::ValidatePackage(speaker_package)) {
      std::cerr << "错误：无效的说话人数据包" << std::endl;
      return 1;
    }

    auto package_info = GPTSoVITS::Utils::SpeakerSerializer::GetPackageInfo(speaker_package);
    if (package_info) {
      std::cout << "  版本: " << package_info->version << std::endl;
      std::cout << "  说话人: " << package_info->speaker_name << std::endl;
      std::cout << "  语言: " << package_info->speaker_lang << std::endl;
      std::cout << "  模型版本: " << package_info->metadata.model_version << std::endl;
    }

    // 加载 G2P Pipeline
    std::cout << "\n[2/5] 加载 G2P Pipeline..." << std::endl;
    auto g2p_pipeline = std::make_shared<GPTSoVITS::G2P::G2PPipline>();

    auto bert_model = std::make_unique<GPTSoVITS::Model::CNBertModel>();
    auto bert_path = modelPath / "bert.onnx";
    auto tokenizer_path = GPTSoVITS::GetGlobalResourcesPath() / "bert_tokenizer.json";
    bert_model->Init<GPTSoVITS::Model::ONNXBackend>(
        bert_path.string(), tokenizer_path.string(), device);

    g2p_pipeline->RegisterLangProcess("zh", std::make_unique<GPTSoVITS::G2P::G2PZH>(),
                                      std::move(bert_model), true);
    g2p_pipeline->RegisterLangProcess("en", std::make_unique<GPTSoVITS::G2P::G2PEN>(),
                                      nullptr, true);
    g2p_pipeline->RegisterLangProcess("ja", std::make_unique<GPTSoVITS::G2P::G2PJA>(),
                                      nullptr, true);
    g2p_pipeline->SetDefaultLang("en");

    // 加载边缘推理所需的模型
    std::cout << "[3/5] 加载边缘推理模型..." << std::endl;
    
    auto gpt_encoder_model = std::make_shared<GPTSoVITS::Model::GPTEncoderModel>();
    gpt_encoder_model->Init<GPTSoVITS::Model::ONNXBackend>(
        (modelPath / "gpt_encoder.onnx").string(), device);

    auto gpt_step_model = std::make_shared<GPTSoVITS::Model::GPTStepModel>();
    gpt_step_model->Init<GPTSoVITS::Model::ONNXBackend>(
        (modelPath / "gpt_step.onnx").string(), device);

    auto sovits_model = std::make_shared<GPTSoVITS::Model::SoVITSModel>();
    sovits_model->Init<GPTSoVITS::Model::ONNXBackend>(
        (modelPath / "sovits.onnx").string(), device);

    // 创建边缘 Pipeline
    std::cout << "[4/5] 创建边缘 Pipeline..." << std::endl;
    auto config_content = readFile((modelPath / "config.json").string());
    auto edge_pipeline = std::make_shared<GPTSoVITS::EdgePipeline>(
        config_content,
        MODEL_PATH,
        g2p_pipeline,
        gpt_encoder_model,
        gpt_step_model,
        sovits_model
    );

    std::cout << "  模型信息:\n" << edge_pipeline->GetModelInfo() << std::endl;

    // 导入说话人数据包
    std::cout << "  导入说话人数据包..." << std::endl;
    if (!edge_pipeline->ImportSpeaker(speaker_package, speaker_name)) {
      std::cerr << "错误：无法导入说话人数据包" << std::endl;
      return 1;
    }

    std::cout << "  已导入说话人: " << speaker_name << std::endl;

    // 执行推理
    std::cout << "\n[5/5] 执行推理..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();

    auto audio = edge_pipeline->InferSpeaker(
        speaker_name, text, text_lang,
        1.0f,    // temperature
        0.5f,    // noise_scale
        1.0f     // speed
    );

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (audio) {
      // 保存音频
      audio->SaveToFile(output_path);

      // 计算统计信息
      int sampling_rate = 32000;  // 从配置获取
      int num_samples = audio->ReadSamples().size();
      double audio_duration_s = static_cast<double>(num_samples) / sampling_rate;

      std::cout << "\n推理统计:" << std::endl;
      std::cout << "  总耗时: " << std::fixed << std::setprecision(2) 
                << elapsed_ms << " ms" << std::endl;
      std::cout << "  音频时长: " << std::fixed << std::setprecision(3) 
                << audio_duration_s << " s" << std::endl;
      std::cout << "  样本数: " << num_samples << std::endl;

      if (audio_duration_s > 0) {
        double rtf = elapsed_ms / 1000.0 / audio_duration_s;
        std::cout << "  实时率 (RTF): " << std::fixed << std::setprecision(4) 
                  << rtf << std::endl;
      }

      std::cout << "\n 音频已保存到: " << output_path << std::endl;

      // 列出所有可用的说话人
      auto speakers = edge_pipeline->ListSpeakers();
      std::cout << "\n可用的说话人:" << std::endl;
      for (const auto& spk : speakers) {
        std::cout << "  - " << spk << std::endl;
      }

    } else {
      std::cerr << "\n 推理失败！" << std::endl;
      return 1;
    }

  } catch (const std::exception& e) {
    std::cerr << "\n 错误: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
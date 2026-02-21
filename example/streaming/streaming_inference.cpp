///
// 流推理示例 - 实时音频生成
//

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/StreamingPipeline.h"
#include "GPTSoVITS/EdgePipeline.h"
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

// 音频分块统计
struct StreamingStats {
  int total_chunks = 0;
  int total_samples = 0;
  double first_packet_latency_ms = 0.0;
  double total_time_ms = 0.0;
  double total_audio_duration_s = 0.0;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point first_packet_time;
  bool first_packet_received = false;
};

// 音频分块处理器
class AudioChunkProcessor {
public:
  AudioChunkProcessor(int sampling_rate) 
      : sampling_rate_(sampling_rate), accumulated_audio_() {}

  void ProcessChunk(const GPTSoVITS::AudioChunk& chunk, StreamingStats& stats) {
    if (!stats.first_packet_received) {
      stats.first_packet_time = std::chrono::steady_clock::now();
      stats.first_packet_latency_ms = std::chrono::duration<double, std::milli>(
          stats.first_packet_time - stats.start_time).count();
      stats.first_packet_received = true;
      
      std::cout << "\n>>> 首包延迟: " << std::fixed << std::setprecision(2) 
                << stats.first_packet_latency_ms << " ms" << std::endl;
    }

    stats.total_chunks++;
    stats.total_samples += chunk.audio_data.size();
    stats.total_audio_duration_s += chunk.duration;

    // 累积音频数据
    accumulated_audio_.insert(accumulated_audio_.end(), 
                             chunk.audio_data.begin(), 
                             chunk.audio_data.end());

    // 打印分块信息
    std::cout << "分块 " << chunk.segment_index << "-" << chunk.chunk_index 
              << " | 时长: " << std::fixed << std::setprecision(3) 
              << chunk.duration << "s | 样本数: " << chunk.audio_data.size() 
              << " | 累计音频: " << std::fixed << std::setprecision(2)
              << stats.total_audio_duration_s << "s" << std::endl;

    // 如果是最后一个分块，保存到文件
    if (chunk.is_last) {
      SaveToFile("streaming_output.wav");
    }
  }

  void SaveToFile(const std::string& filename) {
    if (accumulated_audio_.empty()) {
      std::cerr << "警告：没有音频数据可保存" << std::endl;
      return;
    }

    // 使用 AudioTools 保存 WAV 文件
    auto audio_tools = GPTSoVITS::AudioTools::FromByte(accumulated_audio_, sampling_rate_);
    audio_tools->SaveToFile(filename);
    std::cout << "\n>>> 音频已保存到: " << filename << std::endl;
  }

private:
  int sampling_rate_;
  std::vector<float> accumulated_audio_;
};

int main(int argc, char* argv[]) {
#ifdef _WIN32
  std::system("chcp 65001");
#endif

  try {
    std::cout << "========================================" << std::endl;
    std::cout << "  GPT-SoVITS-CPP 流推理示例" << std::endl;
    std::cout << "========================================" << std::endl;

    // 解析命令行参数
    std::string speaker_package = "firefly.gsppkg";
    std::string text = "你好，这是一段测试文本，用于演示流式推理功能。";
    std::string text_lang = "zh";
    std::string speaker_name = "firefly";

    if (argc >= 2) speaker_package = argv[1];
    if (argc >= 3) text = argv[2];
    if (argc >= 4) text_lang = argv[3];
    if (argc >= 5) speaker_name = argv[4];

    std::filesystem::path modelPath = FS_PATH(MODEL_PATH);

    std::cout << "\n配置信息:" << std::endl;
    std::cout << "  模型路径: " << MODEL_PATH << std::endl;
    std::cout << "  说话人数据包: " << speaker_package << std::endl;
    std::cout << "  文本: " << text << std::endl;
    std::cout << "  语言: " << text_lang << std::endl;
    std::cout << "  设备: " << (device.type == GPTSoVITS::Model::DeviceType::kCUDA ? "CUDA" : "CPU") << std::endl;

    // 加载 G2P Pipeline
    std::cout << "\n[1/4] 加载 G2P Pipeline..." << std::endl;
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

    // 加载边缘推理所需的模型（4个模型，而非完整的8个）
    std::cout << "[2/4] 加载边缘推理模型..." << std::endl;
    
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
    std::cout << "\n[3/4] 导入说话人数据包..." << std::endl;
    if (!edge_pipeline->ImportSpeaker(speaker_package, speaker_name)) {
      std::cerr << "错误：无法导入说话人数据包" << std::endl;
      return 1;
    }

    std::cout << "  已导入说话人: " << speaker_name << std::endl;

    // 配置流推理参数
    GPTSoVITS::StreamingConfig streaming_config;
    streaming_config.chunk_length = 24;      // 分块长度
    streaming_config.pause_length = 0.3f;    // 段落间停顿
    streaming_config.fade_length = 1280;     // 淡入淡出长度
    streaming_config.enable_fade = true;     // 启用淡入淡出

    // 创建流推理 Pipeline
    auto streaming_pipeline = std::make_shared<GPTSoVITS::StreamingPipeline>(
        edge_pipeline, streaming_config
    );

    // 创建音频处理器和统计
    AudioChunkProcessor audio_processor(32000);  // 32kHz
    StreamingStats stats;
    stats.start_time = std::chrono::steady_clock::now();

    // 开始流推理
    std::cout << "\n[4/4] 开始流推理..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    bool success = streaming_pipeline->InferSpeakerStreaming(
        speaker_name,
        text,
        text_lang,
        [&audio_processor, &stats](const GPTSoVITS::AudioChunk& chunk) {
          audio_processor.ProcessChunk(chunk, stats);
        },
        1.0f,    // temperature
        0.5f,    // noise_scale
        1.0f     // speed
    );

    auto end_time = std::chrono::steady_clock::now();
    stats.total_time_ms = std::chrono::duration<double, std::milli>(
        end_time - stats.start_time).count();

    std::cout << "----------------------------------------" << std::endl;

    if (success) {
      // 打印统计信息
      std::cout << "\n流推理统计:" << std::endl;
      std::cout << "  总分块数: " << stats.total_chunks << std::endl;
      std::cout << "  总样本数: " << stats.total_samples << std::endl;
      std::cout << "  首包延迟: " << std::fixed << std::setprecision(2) 
                << stats.first_packet_latency_ms << " ms" << std::endl;
      std::cout << "  总耗时: " << std::fixed << std::setprecision(2) 
                << stats.total_time_ms << " ms" << std::endl;
      std::cout << "  音频时长: " << std::fixed << std::setprecision(3) 
                << stats.total_audio_duration_s << " s" << std::endl;
      
      if (stats.total_audio_duration_s > 0) {
        double rtf = stats.total_time_ms / 1000.0 / stats.total_audio_duration_s;
        std::cout << "  实时率 (RTF): " << std::fixed << std::setprecision(4) 
                  << rtf << std::endl;
      }

      std::cout << "\n 流推理完成！" << std::endl;
    } else {
      std::cerr << "\n 流推理失败！" << std::endl;
      return 1;
    }

  } catch (const std::exception& e) {
    std::cerr << "\n 错误: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
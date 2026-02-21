//
// 云端：创建说话人并导出数据包
//
// 场景：高性能服务器端创建说话人数据，打包后发送到边缘设备
// 优势：边缘设备只需加载4个推理模型，无需SSL/VQ等模型
//

#include <iostream>
#include <memory>
#include <string>

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
    std::cout << "  创建说话人数据包" << std::endl;
    std::cout << "========================================" << std::endl;

    // 解析命令行参数
    std::string speaker_name = "firefly";
    std::string ref_audio_path = R"(./看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物钟表小子.wav)";
    std::string ref_text = "看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物钟表小子";
    std::string ref_lang = "zh";
    std::string output_path = "firefly.gsppkg";

    if (argc >= 2) speaker_name = argv[1];
    if (argc >= 3) ref_audio_path = argv[2];
    if (argc >= 4) ref_text = argv[3];
    if (argc >= 5) ref_lang = argv[4];
    if (argc >= 6) output_path = argv[5];

    std::filesystem::path modelPath = FS_PATH(MODEL_PATH);

    std::cout << "\n配置信息:" << std::endl;
    std::cout << "  说话人名称: " << speaker_name << std::endl;
    std::cout << "  参考音频: " << ref_audio_path << std::endl;
    std::cout << "  参考文本: " << ref_text << std::endl;
    std::cout << "  参考语言: " << ref_lang << std::endl;
    std::cout << "  输出路径: " << output_path << std::endl;
    std::cout << "  设备: " << (device.type == GPTSoVITS::Model::DeviceType::kCUDA ? "CUDA" : "CPU") << std::endl;

    // 加载 G2P Pipeline
    std::cout << "\n[1/5] 加载 G2P Pipeline..." << std::endl;
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

    // 加载完整的模型（8个模型，用于创建说话人）
    std::cout << "[2/5] 加载完整模型..." << std::endl;
    
    auto ssl_model = std::make_shared<GPTSoVITS::Model::SSLModel>();
    ssl_model->Init<GPTSoVITS::Model::ONNXBackend>(
        (modelPath / "ssl.onnx").string(), device);

    auto vq_model = std::make_shared<GPTSoVITS::Model::VQModel>();
    vq_model->Init<GPTSoVITS::Model::ONNXBackend>(
        (modelPath / "vq_encoder.onnx").string(), device);

    auto spectrogram_model = std::make_shared<GPTSoVITS::Model::SpectrogramModel>();
    spectrogram_model->Init<GPTSoVITS::Model::ONNXBackend>(
        (modelPath / "spectrogram.onnx").string(), device);

    auto sv_embedding_model = std::make_shared<GPTSoVITS::Model::SVEmbeddingModel>();
    sv_embedding_model->Init<GPTSoVITS::Model::ONNXBackend>(
        (modelPath / "sv_embedding.onnx").string(), device);

    auto gpt_encoder_model = std::make_shared<GPTSoVITS::Model::GPTEncoderModel>();
    gpt_encoder_model->Init<GPTSoVITS::Model::ONNXBackend>(
        (modelPath / "gpt_encoder.onnx").string(), device);

    auto gpt_step_model = std::make_shared<GPTSoVITS::Model::GPTStepModel>();
    gpt_step_model->Init<GPTSoVITS::Model::ONNXBackend>(
        (modelPath / "gpt_step.onnx").string(), device);

    auto sovits_model = std::make_shared<GPTSoVITS::Model::SoVITSModel>();
    sovits_model->Init<GPTSoVITS::Model::ONNXBackend>(
        (modelPath / "sovits.onnx").string(), device);

    // 创建完整 Pipeline
    std::cout << "[3/5] 创建完整 Pipeline..." << std::endl;
    auto config_content = readFile((modelPath / "config.json").string());
    GPTSoVITS::GPTSoVITSPipline pipeline(
        config_content, g2p_pipeline, ssl_model, vq_model,
        spectrogram_model, sv_embedding_model, gpt_encoder_model,
        gpt_step_model, sovits_model);

    // 创建说话人
    std::cout << "[4/5] 创建说话人..." << std::endl;
    const auto& speaker_info = pipeline.CreateSpeaker(
        speaker_name, ref_lang, FS_PATH(ref_audio_path), ref_text);

    std::cout << "  说话人创建成功: " << speaker_info.SpeakerName() << std::endl;
    std::cout << "  语言: " << speaker_info.SpeakerLang() << std::endl;

    // 导出说话人数据包
    std::cout << "[5/5] 导出说话人数据包..." << std::endl;
    if (!pipeline.ExportSpeaker(speaker_name, output_path)) {
      std::cerr << "错误：无法导出说话人数据包" << std::endl;
      return 1;
    }

    // 获取数据包信息
    auto package_info = GPTSoVITS::Utils::SpeakerSerializer::GetPackageInfo(output_path);
    if (package_info) {
      std::cout << "\n数据包信息:" << std::endl;
      std::cout << "  版本: " << package_info->version << std::endl;
      std::cout << "  说话人: " << package_info->speaker_name << std::endl;
      std::cout << "  语言: " << package_info->speaker_lang << std::endl;
      std::cout << "  模型版本: " << package_info->metadata.model_version << std::endl;
      std::cout << "  SV 维度: " << package_info->metadata.sv_dim << std::endl;
      std::cout << "  最大序列长度: " << package_info->metadata.max_seq_len << std::endl;
      std::cout << "  张量数量: " << package_info->tensors.size() << std::endl;
    }

    std::cout << "\n 说话人数据包已导出: " << output_path << std::endl;
    std::cout << "\n使用方法:" << std::endl;
    std::cout << "  将数据包传输到边缘设备，然后运行:" << std::endl;
    std::cout << "  ./edge_inference " << output_path << " \"你好，这是测试文本。\"" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "\n 错误: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
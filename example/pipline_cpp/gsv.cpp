//
// Created by 19254 on 2026/2/8.
//
#include <iostream>

#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"

#ifdef _HOST_WONDOWS_
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

std::shared_ptr<GPTSoVITS::G2P::G2PPipline> load_g2p_pipline(
    const std::filesystem::path& model_path) {
  auto pipeline = std::make_shared<GPTSoVITS::G2P::G2PPipline>();

  auto bert_model = std::make_unique<GPTSoVITS::Model::CNBertModel>();
  auto bert_path = model_path / "bert.onnx";
  auto tokenizer_path =
      GPTSoVITS::GetGlobalResourcesPath() / "bert_tokenizer.json";
  bert_model->Init<GPTSoVITS::Model::ONNXBackend>(
      bert_path.string(), tokenizer_path.string(), device);

  pipeline->RegisterLangProcess("zh", std::make_unique<GPTSoVITS::G2P::G2PZH>(),
                                std::move(bert_model), true);
  pipeline->RegisterLangProcess("en", std::make_unique<GPTSoVITS::G2P::G2PEN>(),
                                nullptr, true);
  pipeline->RegisterLangProcess("ja", std::make_unique<GPTSoVITS::G2P::G2PJA>(),
                                nullptr, true);
  pipeline->SetDefaultLang("en");
  return pipeline;
}

std::shared_ptr<GPTSoVITS::Model::VQModel> load_vq_model(
    const std::filesystem::path& model_path) {
  auto model = std::make_shared<GPTSoVITS::Model::VQModel>();
  model->Init<GPTSoVITS::Model::ONNXBackend>(
      (model_path / "vq_encoder.onnx").string(), device);
  return model;
}

std::shared_ptr<GPTSoVITS::Model::SSLModel> load_ssl_model(
    const std::filesystem::path& model_path) {
  auto model = std::make_shared<GPTSoVITS::Model::SSLModel>();
  model->Init<GPTSoVITS::Model::ONNXBackend>((model_path / "ssl.onnx").string(),
                                             device);
  return model;
}

std::shared_ptr<GPTSoVITS::Model::GPTEncoderModel> load_gpt_encoder_model(
    const std::filesystem::path& model_path) {
  auto model = std::make_shared<GPTSoVITS::Model::GPTEncoderModel>();
  model->Init<GPTSoVITS::Model::ONNXBackend>(
      (model_path / "gpt_encoder.onnx").string(), device);
  return model;
}

std::shared_ptr<GPTSoVITS::Model::GPTStepModel> load_gpt_step_model(
    const std::filesystem::path& model_path) {
  auto model = std::make_shared<GPTSoVITS::Model::GPTStepModel>();
  model->Init<GPTSoVITS::Model::ONNXBackend>(
      (model_path / "gpt_step.onnx").string(), device);
  return model;
}

std::shared_ptr<GPTSoVITS::Model::SpectrogramModel> load_spectrogram_model(
    const std::filesystem::path& model_path) {
  auto model = std::make_shared<GPTSoVITS::Model::SpectrogramModel>();
  model->Init<GPTSoVITS::Model::ONNXBackend>(
      (model_path / "spectrogram.onnx").string(), device);
  return model;
}

std::shared_ptr<GPTSoVITS::Model::SVEmbeddingModel> load_sv_embedding_model(
    const std::filesystem::path& model_path) {
  auto model = std::make_shared<GPTSoVITS::Model::SVEmbeddingModel>();
  model->Init<GPTSoVITS::Model::ONNXBackend>(
      (model_path / "sv_embedding.onnx").string(), device);
  return model;
}

std::shared_ptr<GPTSoVITS::Model::SoVITSModel> load_sovits_model(
    const std::filesystem::path& model_path) {
  auto model = std::make_shared<GPTSoVITS::Model::SoVITSModel>();
  model->Init<GPTSoVITS::Model::ONNXBackend>(
      (model_path / "sovits.onnx").string(), device);
  return model;
}

int main() {
#ifdef _WIN32
  std::system("chcp 65001");
#endif
  std::filesystem::path modelPath = FS_PATH(MODEL_PATH);

  GPTSoVITS::GPTSoVITSPipline gsv(
      readFile((modelPath / "config.json").string()), load_g2p_pipline(modelPath),
      load_ssl_model(modelPath), load_vq_model(modelPath),
      load_spectrogram_model(modelPath), load_sv_embedding_model(modelPath),
      load_gpt_encoder_model(modelPath), load_gpt_step_model(modelPath),
      load_sovits_model(modelPath));

  gsv.CreateSpeaker(
      "firefly", "zh",
      FS_PATH(
          R"(F:\Engcode\c_c++\huiyicc\gsv_cpp\cmake-build-debug\看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物钟表小子.wav)"),
      "看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物钟表小子");

  // Test inference (currently returns empty audio as SoVITS generation is not
  // yet implemented)
  auto result = gsv.InferSpeaker("firefly", "你好，这是一段测试文本。");
  if (result) {
    result->SaveToFile("output_test.wav");
    std::cout << "Audio saved to output_test.wav" << std::endl;
  }

  return 0;
}

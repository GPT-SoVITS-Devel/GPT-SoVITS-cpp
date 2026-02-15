//
// Created by 19254 on 2026/2/8.
//
#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"

#ifdef _HOST_WONDOWS_
auto device = GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCUDA, 0);
#else
auto device = GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCPU, 0);
#endif


std::shared_ptr<GPTSoVITS::G2P::G2PPipline> load_g2p_pipline(
    const std::filesystem::path& model_path) {
  auto pipeline = std::make_shared<GPTSoVITS::G2P::G2PPipline>();

  auto bert_model = std::make_unique<GPTSoVITS::Model::CNBertModel>();
  auto bert_path = model_path / "bert.onnx";
  auto tokenizer_path =
      GPTSoVITS::GetGlobalResourcesPath() / "bert_tokenizer.json";
  bert_model->Init<GPTSoVITS::Model::ONNXBackend>(
      bert_path.string(), tokenizer_path.string(),
      device);

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
  (model_path / "vq_encoder.onnx").string(),
  device);
  return model;
}
std::shared_ptr<GPTSoVITS::Model::SSLModel> load_ssl_model(
    const std::filesystem::path& model_path) {
  auto model = std::make_shared<GPTSoVITS::Model::SSLModel>();
  model->Init<GPTSoVITS::Model::ONNXBackend>(
  (model_path / "ssl.onnx").string(),
  device);
  return model;
}

int main() {
#ifdef _WIN32
  std::system("chcp 65001");
#endif
  std::filesystem::path modelPath =
      R"(/Users/huiyi/code/python/GPT-SoVITS_minimal_inference/onnx_export/firefly_v2_proplus_fp16)";

  GPTSoVITS::GPTSoVITSPipline gsv(load_g2p_pipline(modelPath),
                                  load_ssl_model(modelPath),
                                  load_vq_model(modelPath));

  gsv.CreateSpeaker(
      "firefly", "zh",
      FS_PATH(R"(/Users/huiyi/code/python/GPT-SoVITS_minimal_inference/pretrained_models/看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物钟表小子.wav)").string(),
      "看，这尊雕像就是匹诺康尼大名鼎鼎的卡通人物钟表小子");
}

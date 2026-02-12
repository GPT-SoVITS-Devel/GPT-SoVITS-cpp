//
// Created by 19254 on 2026/2/8.
//
#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"

std::shared_ptr<GPTSoVITS::G2P::G2PPipline> load_g2p_pipline(
    const std::filesystem::path& model_path) {
  auto pipeline = std::make_shared<GPTSoVITS::G2P::G2PPipline>();

  auto bert_model = std::make_unique<GPTSoVITS::Model::CNBertModel>();
  auto bert_path = model_path / "bert.onnx";
  auto tokenizer_path =
      GPTSoVITS::GetGlobalResourcesPath() / "bert_tokenizer.json";
  bert_model->Init<GPTSoVITS::Model::ONNXBackend>(
      bert_path.string(), tokenizer_path.string(),
      GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCUDA, 0));

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
  GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCUDA, 0));
  return model;
}
std::shared_ptr<GPTSoVITS::Model::SSLModel> load_ssl_model(
    const std::filesystem::path& model_path) {
  auto model = std::make_shared<GPTSoVITS::Model::SSLModel>();
  model->Init<GPTSoVITS::Model::ONNXBackend>(
  (model_path / "ssl.onnx").string(),
  GPTSoVITS::Model::Device(GPTSoVITS::Model::DeviceType::kCUDA, 0));
  return model;
}

int main() {
#ifdef _WIN32
  std::system("chcp 65001");
#endif
  std::filesystem::path modelPath =
      R"(F:\Engcode\AIAssistant\GPT-SoVITS-Devel\GPT-SoVITS_minimal_inference\onnx_export\firefly_v2_proplus_fp16)";

  GPTSoVITS::GPTSoVITSPipline gsv(load_g2p_pipline(modelPath),
                                  load_ssl_model(modelPath),
                                  load_vq_model(modelPath));

  gsv.CreateSpeaker(
      "firefly", "zh",
      FS_PATH(R"(G:\dataset\audio\mihoyo\gsv\firefly\虽然我也没太搞清楚状况…但他说，似乎只有直率、纯真、有童心的小孩子才能看见它…….wav)").string(),
      "虽然我也没太搞清楚状况…但他说，似乎只有直率、纯真、有童心的小孩子才能看"
      "见它……");
}

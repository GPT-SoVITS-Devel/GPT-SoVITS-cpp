//
// Created by 19254 on 2026/1/16.
//
#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"
#include "GPTSoVITS/plog.h"

int main() {
  GPTSoVITS::Model::BertModel bertModel;
  bertModel.Init<GPTSoVITS::Model::ONNXBackend>(
      R"(F:\Engcode\AIAssistant\GPT-SoVITS-Devel\GPT-SoVITS_minimal_inference\onnx_export\firefly_v2_proplus_fp16\bert.onnx)",
      GPTSoVITS::Model::Device{GPTSoVITS::Model::DeviceType::kCUDA}, 2);
  // auto g2p = GPTSoVITS::G2P::MakeFromLang("zh");
  // auto res = g2p->CleanText("今天天气还不错~");

  return 0;
}
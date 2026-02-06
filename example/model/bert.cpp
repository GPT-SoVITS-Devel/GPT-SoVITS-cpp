#include <filesystem>
#include "GPTSoVITS/G2P/Pipline.h"
#include "GPTSoVITS/G2P/g2p_zh.h"
#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/Text/Sentence.h"
#include "GPTSoVITS/Text/Coding.h"
#include "GPTSoVITS/model/CNBertModel.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS {
extern std::filesystem::path g_globalResourcesPath;
}

int main() {
  using namespace GPTSoVITS::Model;
  using namespace GPTSoVITS::G2P;
  using namespace GPTSoVITS::Text;

  GPTSoVITS::g_globalResourcesPath = R"(F:\Engcode\c_c++\huiyicc\gsv_cpp\res)";

  auto pipeline = std::make_shared<G2PPipline>();

  auto zh_g2p = std::make_unique<G2PZH>();
  auto bert_model = std::make_unique<CNBertModel>();

  std::string bert_path = R"(F:\Engcode\AIAssistant\GPT-SoVITS-Devel\GPT-SoVITS_minimal_inference\onnx_export\firefly_v2_proplus\bert.onnx)";
  std::string tokenizer_path = (GPTSoVITS::g_globalResourcesPath / "tokenizer_many_lang.json").string();

  try {
      // ONNX 后端 + CUDA
      PrintInfo("Loading BERT model into memory (Device: CUDA:0)...");
      bert_model->Init<ONNXBackend>(bert_path, tokenizer_path, Device(DeviceType::kCUDA, 0));

      pipeline->RegisterLangProcess("zh", std::move(zh_g2p), std::move(bert_model));
      pipeline->SetDefaultLang("zh");

      // Punctuation 模式
      Sentence sentence(Sentence::SentenceSplitMethod::Punctuation);
      
      // 设置切分回调: 每当切分出一个完整句子时, 触发 G2P + BERT 推理
      sentence.AppendCallBack([pipeline](const std::string &seg) -> bool {
          PrintInfo(">>> [Segment Split] Processing: {}", seg);
          try {
              // 推理流: Text -> G2P (Phonemes) -> BERT (Features)
              auto res = pipeline->GetPhoneAndBert(seg);
              
              if (res && res->BertSeq && res->PhoneSeq) {
                  int64_t phone_len = res->PhoneSeq->Shape()[0];
                  int64_t bert_dim = res->BertSeq->Shape()[0];
                  int64_t bert_len = res->BertSeq->Shape()[1];

                  PrintInfo("    [Inference Success] Phones: {}, Bert Shape: {}x{}", 
                            phone_len, bert_dim, bert_len);

                  // 打印音素 ID 样例
                  std::string ids = "";
                  for(int i=0; i<std::min<int>(phone_len, 10); ++i) 
                      ids += std::to_string(res->PhoneSeq->At<int64_t>(i)) + " ";
                  PrintInfo("    [Data Sample] First 10 Phone IDs: {}...", ids);


              }
          } catch (const std::exception &e) {
              PrintError("    [Inference Error] Failed to process segment: {}", e.what());
          }
          return true;
      });

      std::string test_strs = 
          "大家好，这里是高性能推理库。它旨在提供一个轻量级、生产就绪的替代方案，"
          "适用于各种需要低延迟语音合成的场景。目前我们正在测试中文流水线，"
          "确保文本正规化、分词以及特征提取都能稳定运行。";

      PrintInfo("Starting streaming text input simulation...");
      
      // 仿照 clean_text.cpp 的 chunked 输入方式
      auto u32t = StringToU32String(test_strs);
      size_t index = 0;
      size_t step = 15; // 每次输入 15 个字符, 模拟打字或流式 ASR 输出

      while (index < u32t.size()) {
          std::string chunk = U32StringToString(u32t.substr(index, step));
          PrintDebug("Feeding text chunk: {}", chunk);
          sentence.Append(chunk);
          index += step;
      }

      sentence.Flush();

      PrintInfo("Streaming simulation completed.");

  } catch (const std::exception& e) {
      PrintError("CRITICAL: Pipeline initialization failed: {}", e.what());
  }

  return 0;
}

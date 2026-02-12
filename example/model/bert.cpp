#include <filesystem>

#include "GPTSoVITS/G2P/G2P_EN.h"
#include "GPTSoVITS/G2P/G2P_JA.h"
#include "GPTSoVITS/G2P/G2P_Zh.h"
#include "GPTSoVITS/G2P/Pipline.h"
#include "GPTSoVITS/GPTSoVITS.h"
#include "GPTSoVITS/Text/Coding.h"
#include "GPTSoVITS/Text/Sentence.h"
#include "GPTSoVITS/model/CNBertModel.h"
#include "GPTSoVITS/model/backend/onnx_backend.h"
#include "GPTSoVITS/plog.h"

namespace GPTSoVITS {
extern std::filesystem::path g_globalResourcesPath;
}

int main() {
#ifdef _WIN32
  std::system("chcp 65001");
#endif
  using namespace GPTSoVITS::Model;
  using namespace GPTSoVITS::G2P;
  using namespace GPTSoVITS::Text;

  auto pipeline = std::make_shared<G2PPipline>();

  auto bert_model = std::make_unique<CNBertModel>();

  std::string bert_path =
      R"(F:\Engcode\AIAssistant\GPT-SoVITS-Devel\GPT-SoVITS_minimal_inference\onnx_export\firefly_v2_proplus_fp16\bert.onnx)";

  std::string tokenizer_path =
      (GPTSoVITS::g_globalResourcesPath / "bert_tokenizer.json").string();

  try {
    // ONNX 后端 + CUDA
    PrintInfo("Loading BERT model into memory (Device: CUDA:0)...");
    bert_model->Init<ONNXBackend>(bert_path, tokenizer_path,
                                  Device(DeviceType::kCUDA, 0));

    pipeline->RegisterLangProcess("zh", std::make_unique<G2PZH>(),
                                  std::move(bert_model),true);
    pipeline->RegisterLangProcess("en", std::make_unique<G2PEN>(), nullptr,true);
    pipeline->RegisterLangProcess("ja", std::make_unique<G2PJA>(), nullptr,true);
    pipeline->SetDefaultLang("en");

    // Punctuation 模式
    Sentence sentence;

    // 设置切分回调: 每当切分出一个完整句子时, 触发 G2P + BERT 推理
    sentence.AppendCallBack([&pipeline](const std::string& seg) -> bool {
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
          for (int i = 0; i < std::min<int>(phone_len, 10); ++i)
            ids += std::to_string(res->PhoneSeq->At<int64_t>(i)) + " ";
          PrintInfo("    [Data Sample] First 10 Phone IDs: {}...", ids);
        }
      } catch (const std::exception& e) {
        PrintError("    [Inference Error] Failed to process segment: {}",
                   e.what());
      }
      return true;
    });

    std::string test_strs =
        "皆さん、我在インターネット上看到someone把几国language混在一起speak。我看到之后be like：それは我じゃないか！私もtry一tryです。\n"
        "虽然是混乱している句子ですけど、中文日本語プラスEnglish、挑戦スタート！\n"
        "我study日本語的时候，もし有汉字，我会很happy。\n"
        "Bueause中国人として、when I see汉字，すぐに那个汉字がわかります。\n"
        "But 我hate外来語、什么マクドナルド、スターバックス、グーグル、ディズニーランド、根本记不住カタカナhow to写、太難しい。\n"
        "2021年6月25日,今天32°C。以上です，byebye！";

    PrintInfo("Starting streaming text input simulation...");

    auto u32t = StringToU32String(test_strs);
    size_t index = 0;
    size_t step = 15;  // 每次输入 15 个字符, 模拟打字或流式 ASR 输出

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

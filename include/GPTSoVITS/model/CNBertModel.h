//
// Created by 19254 on 2026/2/5.
//

#ifndef GSV_CPP_CNBERTMODEL_H
#define GSV_CPP_CNBERTMODEL_H
#include <GPTSoVITS/model/bert.h>

#include <fstream>
#include <memory>

#include "GPTSoVITS/Utils/exception.h"
#include "tokenizers_cpp.h"

namespace GPTSoVITS::Model {

class CNBertModel : public BertModel {
  std::unique_ptr<tokenizers::Tokenizer> m_tokenzer;

public:
  CNBertModel();

  ~CNBertModel() = default;

  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path, const std::string& tokenzer_path,
            const Device& device = DeviceType::kCPU, int work_thread_num = 1) {
    std::ifstream file(tokenzer_path.data());
    if (!file.is_open()) {
      THROW_ERRORN("加载Tokenizer失败\nBy:{}", tokenzer_path.data());
    }
    std::string content;
    file >> content;
    m_tokenzer = tokenizers::Tokenizer::FromBlobJSON(content);
    BertModel::Init<MODEL_BACKEND>(model_path, device, work_thread_num);
  };

  EncodeResult EncodeText(const std::string& text);
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_CNBERTMODEL_H

//
// Created by 19254 on 2026/1/16.
//

#ifndef GSV_CPP_MODEL_BERT_H
#define GSV_CPP_MODEL_BERT_H

#include "GPTSoVITS/model/base.h"
#include "GPTSoVITS/model/tensor.h"

namespace GPTSoVITS::Model {

class BertModel {
protected:
  std::unique_ptr<BaseModel> m_model;

public:
  explicit BertModel() = default;

  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path,
            const Device& device = DeviceType::kCPU, int work_thread_num = 1) {
    m_model = std::make_unique<MODEL_BACKEND>();
    m_model->Load(model_path, device, work_thread_num);
  }

  struct EncodeResult {
    std::vector<int> TokenIds;
    std::vector<int> TokenTypeIds;
    std::vector<int> Masks;
  };

  EncodeResult EncodeText(const std::string& text);
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_MODEL_BERT_H

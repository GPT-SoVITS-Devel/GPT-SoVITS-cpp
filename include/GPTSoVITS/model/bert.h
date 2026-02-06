#ifndef GSV_CPP_MODEL_BERT_H
#define GSV_CPP_MODEL_BERT_H

#include <memory>
#include <string>
#include <vector>

#include "GPTSoVITS/model/base.h"
#include "GPTSoVITS/model/tensor.h"
#include "GPTSoVITS/G2P/Base.h"

namespace GPTSoVITS::Bert {

struct BertRes {
  std::shared_ptr<Model::Tensor> PhoneSeq;
  std::shared_ptr<Model::Tensor> BertSeq;
};
}  // namespace GPTSoVITS::Bert

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
    std::vector<int64_t> TokenIds;
    std::vector<int64_t> TokenTypeIds;
    std::vector<int64_t> Masks;
  };

  /**
   * @brief 执行BERT推理并根据word2ph重复特征
   * @param text 规范化后的文本
   * @param g2p_info g2p的数据
   * @return BERT特征张量 (1024, seq_len)
   */
  virtual std::unique_ptr<Tensor> GetBertFeature(const std::string& text,
                                                 const G2P::G2PRes& g2p_info) = 0;

  /**
   * @brief 纯文本编码
   */
  virtual EncodeResult EncodeText(const std::string& text) = 0;
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_MODEL_BERT_H
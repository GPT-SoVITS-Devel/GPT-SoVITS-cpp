//
// Created by Huiyicc on 2026/2/8.
//

#ifndef GSV_CPP_SSL_H
#define GSV_CPP_SSL_H

#include <fstream>

#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/model/base.h"

namespace GPTSoVITS::Model {

class SSLModel {
protected:
  std::unique_ptr<BaseModel> m_model;
public:
  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path,
            const Device& device = DeviceType::kCPU, int work_thread_num = 1) {
    m_model = std::make_unique<MODEL_BACKEND>();
    m_model->Load(model_path, device, work_thread_num);
  };

  std::unique_ptr<Tensor> GetSSLContent(const std::vector<float>& audio_16k);
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_SSL_H

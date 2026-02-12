//
// Created by 19254 on 2026/2/10.
//

#ifndef GSV_CPP_VQ_H
#define GSV_CPP_VQ_H

#include <fstream>

#include "GPTSoVITS/Utils/exception.h"
#include "GPTSoVITS/model/base.h"

namespace GPTSoVITS::Model {

class VQModel {
protected:
  std::unique_ptr<BaseModel> m_model;

public:
  template <typename MODEL_BACKEND>
  void Init(const std::string& model_path,
            const Device& device = DeviceType::kCPU, int work_thread_num = 1) {
    m_model = std::make_unique<MODEL_BACKEND>();
    m_model->Load(model_path, device, work_thread_num);
  };

  std::unique_ptr<Tensor> GetVQCodes(Tensor* ssl_content);

  [[nodiscard]] const BaseModel* GetModel() const { return m_model.get(); }
};

}  // namespace GPTSoVITS::Model

#endif  // GSV_CPP_VQ_H

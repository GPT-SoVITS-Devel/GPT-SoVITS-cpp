//
// Created by 19254 on 2026/2/5.
//
#include <GPTSoVITS/model/CNBertModel.h>

#include <fstream>

#include "GPTSoVITS/Utils/exception.h"

namespace GPTSoVITS::Model {

CNBertModel::CNBertModel() = default;

BertModel::EncodeResult CNBertModel::EncodeText(const std::string& text) {
  auto rust_res = m_tokenzer->EncodeEx(text, true);
  m_model->Forward();

  // EncodeResult res;
  // res.TokenIds = std::move(rust_res.TokenIds);
  // res.TokenTypeIds = std::move(rust_res.TokenTypeIds);
  // res.Masks = std::move(rust_res.Masks);
  // return res;
}

}  // namespace GPTSoVITS::Model
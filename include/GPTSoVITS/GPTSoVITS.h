//
// Created by Huiyicc on 2026/1/12.
//

#ifndef GPT_SOVITS_CPP_GPTSOVITS_H
#define GPT_SOVITS_CPP_GPTSOVITS_H

#include <filesystem>
#ifdef _USE_U8PATH_
#define FS_PATH std::filesystem::u8path
#else
#define FS_PATH std::filesystem::path
#endif

#include "GPTSoVITS/model/tensor.h"
#include "GPTSoVITS/model/device.h"
#include "GPTSoVITS/Text/Coding.h"
#include "GPTSoVITS/Text/LangDetect.h"
#include "GPTSoVITS/Text/Sentence.h"
#include "GPTSoVITS/model/bert.h"
#include "GPTSoVITS/model/vq.h"
#include "GPTSoVITS/model/ssl.h"
#include "GPTSoVITS/model/CNBertModel.h"
#include "GPTSoVITS/G2P/Base.h"
#include "GPTSoVITS/G2P/G2P_EN.h"
#include "GPTSoVITS/G2P/G2P_Zh.h"
#include "GPTSoVITS/G2P/G2P_JA.h"
#include "GPTSoVITS/G2P/Pipline.h"
#include "GPTSoVITS/AudioTools.h"
#include "GPTSoVITS/GPTSoVITSCpp.h"

#endif  // GPT_SOVITS_CPP_GPTSOVITS_H

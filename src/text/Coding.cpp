//
// Created by 19254 on 24-11-29.
//
#include <GPTSoVITS/Text/Coding.h>

#include "utf8.h"

namespace GPTSoVITS::Text {

std::u32string StringToU32String(const std::string &text) {
  std::u32string out;
  utf8::utf8to32(text.begin(), text.end(), std::back_inserter(out));
  return out;
};

std::string U32StringToString(const std::u32string &text) {
  std::string out;
  utf8::utf32to8(text.begin(), text.end(), std::back_inserter(out));
  return out;
}

std::wstring Utf8ToWstring(const std::string& utf8_str) {
  if (utf8_str.empty()) {
    return L"";
  }

  std::wstring result;
  if constexpr (sizeof(wchar_t) == 2) {
    // Windows 环境: wchar_t 为 UTF-16
    utf8::utf8to16(utf8_str.begin(), utf8_str.end(), std::back_inserter(result));
  } else {
    // Unix/Linux 环境: wchar_t 为 UTF-32
    utf8::utf8to32(utf8_str.begin(), utf8_str.end(), std::back_inserter(result));
  }

  return result;
}

std::wstring Utf8ToWstringSafe(const std::string& utf8_str) {
  // 检查无效的 UTF-8 序列
  if (!utf8::is_valid(utf8_str.begin(), utf8_str.end())) {
    return L"";
  }
  return Utf8ToWstring(utf8_str);
}

}
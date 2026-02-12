//
// Created by Huiyicc on 24-11-29.
//

#ifndef GPT_SOVITS_CPP_CODING_H
#define GPT_SOVITS_CPP_CODING_H

#include <string>

namespace GPTSoVITS::Text {

std::u32string StringToU32String(const std::string&text);
std::string U32StringToString(const std::u32string &text);

/**
 * @brief 将 UTF-8 编码的 std::string 转换为 std::wstring
 * @param utf8_str 输入的 UTF-8 字符串
 * @return 转换后的 std::wstring
 * @throws std::runtime_error 当检测到无效的 UTF-8 序列时抛出
 */
std::wstring Utf8ToWstring(const std::string& utf8_str);

/**
 * @brief 安全版本的转换，处理无效字节序而不崩溃
 */
std::wstring Utf8ToWstringSafe(const std::string& utf8_str);

}

#endif // GPT_SOVITS_CPP_CODING_H

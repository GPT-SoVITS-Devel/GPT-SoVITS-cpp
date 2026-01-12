//
// Created by 19254 on 24-11-10.
//
#include <regex>

#include "GPTSoVITS/Text/TextNormalizerMap.h"

namespace GPTSoVITS::Text {

std::u32string process_with_regex(const std::u32string &text, const srell::u32regex &re,
                                  const std::function<std::u32string(const srell::u32smatch &)> &replace_func) {
  std::u32string result;
  std::u32string::const_iterator searchStart(text.cbegin());
  srell::u32smatch match;

  while (srell::regex_search(searchStart, text.cend(), match, re)) {
    // Append text before the match
    result.append(searchStart, match[0].first);
    // Append the replaced text
    result.append(replace_func(match));

    // Update the search start position
    searchStart = match.suffix().first;
  }
  // Append the remaining part of the text
  result.append(searchStart, text.cend());

  return result;
}
}
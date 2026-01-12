//
// Created by 19254 on 2026/1/13.
//
#include "GPTSoVITS/GPTSoVITS.h"
#include <filesystem>


namespace GPTSoVITS {
std::filesystem::path g_globalResourcesPath = std::filesystem::current_path() / "res";
}
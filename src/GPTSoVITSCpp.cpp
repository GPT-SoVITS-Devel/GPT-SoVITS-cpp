//
// Created by Huiyicc on 2026/1/17.
//
#include "GPTSoVITS/GPTSoVITSCpp.h"

namespace GPTSoVITS {

std::filesystem::path g_globalResourcesPath =
    std::filesystem::current_path() / "res";

void SetGlobalResourcesPath(const std::string& path) {
  g_globalResourcesPath = path;
}
std::filesystem::path GetGlobalResourcesPath() { return g_globalResourcesPath; }

const SpeakerInfo& GPTSoVITSPipline::CreateSpeaker(
    const std::string& speaker_name, const std::string& ref_audio_lang,
    const std::string& ref_audio_path, const std::string& ref_audio_text) {
  auto iter = m_speaker_map.find(speaker_name);
  if (iter != m_speaker_map.end()) {
    return iter->second;
  }
  SpeakerInfo info;
  auto refAudio = AudioTools::FromFile(ref_audio_path);
  info.m_speaker_16k = refAudio->ReSample(16000);
  info.m_speaker_lang = ref_audio_lang;
  info.m_ssl_content = m_ssl_model->GetSSLContent(info.m_speaker_16k->ReadSamples());
  info.m_vq_codes = m_vq_model->GetVQCodes(info.m_ssl_content.get());
  // TODO: 要删除第一个维度
  m_speaker_map[speaker_name] = std::move(info);
  return m_speaker_map[speaker_name];
}

}  // namespace GPTSoVITS

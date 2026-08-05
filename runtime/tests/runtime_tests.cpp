#include "dark_preview_denoise.h"

#include <array>
#include <cassert>
#include <cstdint>

int main() {
  using namespace ai_isp;
  const auto& profile = GetFixedRyybProfile();
  assert(profile.valid_width == 1024 && profile.valid_height == 768);
  AdmissionPolicy policy{"main_ryyb_0", "tele_ryyb_0", RyybCfaPhase::kRyyb,
                         RyybCfaPhase::kByyr, "model", "quant"};
  RyybFrameDescriptor frame{};
  frame.camera = CameraId::kMain;
  frame.sensor_profile = "main_ryyb_0";
  frame.raw_width = 2048;
  frame.raw_height = 1536;
  frame.crop_width = 2048;
  frame.crop_height = 1536;
  frame.row_stride_bytes = 4096;
  frame.bit_depth = 12;
  frame.white_level = {4095.0F, 4095.0F, 4095.0F, 4095.0F};
  frame.model_hash = "model";
  frame.quant_policy_hash = "quant";
  assert(ValidateAiAdmission(frame, policy) == Status::kOk);
  frame.crop_x = 1;
  assert(ValidateAiAdmission(frame, policy) == Status::kInvalidCfaPhase);
  frame.crop_x = 0;
  frame.camera = CameraId::kTele;
  frame.sensor_profile = "tele_ryyb_0";
  assert(ValidateAiAdmission(frame, policy) == Status::kInvalidCfaPhase);
  frame.cfa_phase = RyybCfaPhase::kByyr;
  assert(ValidateAiAdmission(frame, policy) == Status::kOk);
  frame.camera = CameraId::kUltrawide;
  assert(ValidateAiAdmission(frame, policy) == Status::kUnsupportedCamera);

  std::array<std::uint16_t, 4 * 8 * 8> input{};
  std::array<std::uint16_t, 4 * 8 * 8> output{};
  for (std::size_t index = 0; index < input.size(); ++index) input[index] = static_cast<std::uint16_t>(index);
  PackedRawView source{input.data(), 8, 8, 16, 128};
  MutablePackedRawView destination{output.data(), 8, 8, 16, 128};
  assert(BitExactBypass(source, destination) == Status::kOk);
  assert(input == output);

  DarkTrigger trigger;
  FrameMetadata dark{};
  dark.metadata_valid = true;
  dark.iso = 1600;
  dark.scene_ev = -2.0F;
  dark.noise_level = 0.09F;
  assert(trigger.Update(dark).bypass);
  assert(trigger.Update(dark).bypass);
  const auto active_ramp = trigger.Update(dark);
  assert(!active_ramp.bypass && active_ramp.enhancement_strength == 0.20F);
  assert(trigger.FailImmediately().bypass);

  UnavailableNpuExecutor executor;
  assert(executor.Load() == Status::kNpuUnavailable);
  return 0;
}

#include "dark_preview_denoise.h"

#include <array>
#include <cassert>
#include <cstdint>

int main() {
  using namespace ai_isp;
  assert(SelectProfile(1024, 768)->id == ProfileId::kP0);
  assert(SelectProfile(960, 540)->compile_height == 544);
  assert(SelectProfile(540, 960) == nullptr);

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
  assert(executor.Load(ProfileId::kP0) == Status::kNpuUnavailable);
  return 0;
}


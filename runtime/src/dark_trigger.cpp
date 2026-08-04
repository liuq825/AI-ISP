#include "dark_preview_denoise.h"

#include <algorithm>
#include <array>

namespace ai_isp {
namespace {
constexpr std::array<float, 5> kRamp{{0.20F, 0.40F, 0.60F, 0.80F, 1.00F}};
}  // namespace

TriggerDecision DarkTrigger::Update(const FrameMetadata& metadata) {
  if (state_ == State::kBypassError || !metadata.metadata_valid ||
      metadata.camera_transition || metadata.thermal_bypass) {
    return metadata.metadata_valid && !metadata.camera_transition && !metadata.thermal_bypass
               ? TriggerDecision{true, 0.0F}
               : FailImmediately();
  }
  const bool enter =
      (metadata.dark_score.has_value() && metadata.dark_score.value() >= 0.70F) ||
      (metadata.iso >= 1600.0F && metadata.scene_ev <= -1.5F) ||
      metadata.noise_level >= 0.08F;
  const bool exit =
      (!metadata.dark_score.has_value() || metadata.dark_score.value() <= 0.45F) &&
      metadata.iso <= 1200.0F && metadata.scene_ev >= -1.0F &&
      metadata.noise_level <= 0.06F;

  if (state_ == State::kBypassBright || state_ == State::kArming) {
    if (!enter) {
      state_ = State::kBypassBright;
      enter_counter_ = 0;
      return {true, 0.0F};
    }
    state_ = State::kArming;
    if (++enter_counter_ >= 3) {
      state_ = State::kActiveRamp;
      ramp_index_ = 0;
      return {false, kRamp[0]};
    }
    return {true, 0.0F};
  }
  if (state_ == State::kActiveRamp) {
    ramp_index_ = std::min<std::uint32_t>(ramp_index_ + 1, kRamp.size());
    if (ramp_index_ >= kRamp.size()) {
      state_ = State::kActive;
      return {false, 1.0F};
    }
    return {false, kRamp[ramp_index_]};
  }
  if (state_ == State::kActive || state_ == State::kExitPending) {
    if (!exit) {
      state_ = State::kActive;
      exit_counter_ = 0;
      return {false, 1.0F};
    }
    state_ = State::kExitPending;
    ++exit_counter_;
    if (exit_counter_ >= 10) {
      Recover();
      return {true, 0.0F};
    }
    const auto index = std::min<std::uint32_t>(exit_counter_ - 1, kRamp.size() - 1);
    return {false, kRamp[kRamp.size() - 1 - index]};
  }
  return {true, 0.0F};
}

TriggerDecision DarkTrigger::FailImmediately() {
  state_ = State::kBypassError;
  enter_counter_ = exit_counter_ = ramp_index_ = 0;
  return {true, 0.0F};
}

void DarkTrigger::Recover() {
  state_ = State::kBypassBright;
  enter_counter_ = exit_counter_ = ramp_index_ = 0;
}

}  // namespace ai_isp


#include "dark_preview_denoise.h"

#include <array>
#include <cassert>
#include <cstdint>

namespace {

void TestAllCfaRoundTrips() {
  using namespace ai_isp;
  constexpr std::size_t kRawWidth = 10;
  constexpr std::size_t kRawHeight = 8;
  constexpr std::size_t kRawStridePixels = 12;
  constexpr std::size_t kPackedWidth = kRawWidth / 2;
  constexpr std::size_t kPackedHeight = kRawHeight / 2;
  constexpr std::size_t kPackedStridePixels = 8;
  constexpr std::size_t kPlanePixels = kPackedStridePixels * kPackedHeight;

  std::array<std::uint16_t, kRawStridePixels * kRawHeight> physical{};
  std::array<std::uint16_t, kRawStridePixels * kRawHeight> restored{};
  std::array<std::uint16_t, 4 * kPlanePixels> packed{};
  for (std::size_t row = 0; row < kRawHeight; ++row) {
    for (std::size_t column = 0; column < kRawWidth; ++column) {
      physical[row * kRawStridePixels + column] =
          static_cast<std::uint16_t>(row * 100U + column);
    }
  }
  PhysicalRawView raw{physical.data(), kRawWidth, kRawHeight,
                      kRawStridePixels * sizeof(std::uint16_t)};
  MutablePackedRawView semantic{packed.data(), kPackedWidth, kPackedHeight,
                                kPackedStridePixels * sizeof(std::uint16_t),
                                kPlanePixels * sizeof(std::uint16_t)};
  PackedRawView semantic_const{packed.data(), kPackedWidth, kPackedHeight,
                               kPackedStridePixels * sizeof(std::uint16_t),
                               kPlanePixels * sizeof(std::uint16_t)};
  MutablePhysicalRawView raw_restored{restored.data(), kRawWidth, kRawHeight,
                                      kRawStridePixels * sizeof(std::uint16_t)};
  const std::array<RyybCfaPhase, 4> phases{{RyybCfaPhase::kRyyb, RyybCfaPhase::kByyr,
                                           RyybCfaPhase::kYryb, RyybCfaPhase::kYbyr}};
  for (const auto phase : phases) {
    packed.fill(0);
    restored.fill(0);
    assert(ReferencePackRyyb(raw, phase, semantic) == Status::kOk);
    assert(ReferenceUnpackRyyb(semantic_const, phase, raw_restored) == Status::kOk);
    for (std::size_t row = 0; row < kRawHeight; ++row) {
      for (std::size_t column = 0; column < kRawWidth; ++column) {
        assert(restored[row * kRawStridePixels + column] ==
               physical[row * kRawStridePixels + column]);
      }
    }
  }
}

void TestDmaBufLifecycle() {
  using namespace ai_isp;
  DmaBufPoolContract pool;
  assert(pool.ImportOnce(2, 41, 4096) == Status::kOk);
  assert(pool.ImportOnce(2, 41, 4096) == Status::kOk);
  assert(pool.ImportOnce(2, 42, 4096) == Status::kBufferContractMismatch);
  DmaBufFrame frame{};
  frame.buffer_index = 2;
  frame.fd = 41;
  frame.size_bytes = 4096;
  frame.row_stride_bytes = 32;
  frame.valid_width = 16;
  frame.valid_height = 16;
  frame.producer_fence = 7;
  assert(pool.Submit(frame) == Status::kOk);
  assert(pool.Submit(frame) == Status::kBufferBusy);
  assert(pool.SignalConsumerReady(2, 11) == Status::kOk);
  assert(pool.Release(2, 10) == Status::kFenceOrderError);
  assert(pool.Release(2, 11) == Status::kOk);
  assert(pool.Submit(frame) == Status::kFenceOrderError);
  frame.producer_fence = 12;
  frame.cpu_memcpy_bytes = 2;
  assert(pool.Submit(frame) == Status::kBufferContractMismatch);
  frame.cpu_memcpy_bytes = 0;
  assert(pool.Submit(frame) == Status::kOk);
  assert(pool.RecoverTimeout(2) == Status::kOk);
  const auto audit = pool.Audit();
  assert(audit.imported_buffers == 1 && audit.submitted_frames == 2);
  assert(audit.timeout_recoveries == 1 && audit.extra_cpu_memcpy_bytes == 0);
}

}  // namespace

int main() {
  using namespace ai_isp;
  const auto& profile = GetFixedRyybProfile();
  assert(profile.valid_width == 1024 && profile.valid_height == 768);

  AdmissionPolicy policy{};
  policy.main_sensor_profile = "main_ryyb_0";
  policy.tele_sensor_profile = "tele_ryyb_0";
  policy.main_cfa_phase = RyybCfaPhase::kRyyb;
  policy.tele_cfa_phase = RyybCfaPhase::kByyr;
  policy.model_hash = "model";
  policy.quant_policy_hash = "quant";
  policy.raw_domain_profile_hash = "raw-domain";
  policy.main_lsc_profile_hash = "main-lsc";
  policy.tele_lsc_profile_hash = "tele-lsc";
  policy.main_unpack_profile_hash = "main-unpack";
  policy.tele_unpack_profile_hash = "tele-unpack";

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
  frame.raw_domain_state = RawDomainState::kLinearPostBlcLscPreDgain;
  frame.blc_applied = true;
  frame.lsc_applied = true;
  frame.raw_domain_profile_hash = "raw-domain";
  frame.lsc_profile_hash = "main-lsc";
  frame.unpack_profile_hash = "main-unpack";
  frame.buffer_contract_version = "v1";
  frame.buffer_fd = 40;
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
  frame.lsc_profile_hash = "tele-lsc";
  frame.unpack_profile_hash = "tele-unpack";
  assert(ValidateAiAdmission(frame, policy) == Status::kOk);
  frame.extra_cpu_memcpy_bytes = 2;
  assert(ValidateAiAdmission(frame, policy) == Status::kBufferContractMismatch);
  frame.extra_cpu_memcpy_bytes = 0;
  frame.camera = CameraId::kUltrawide;
  assert(ValidateAiAdmission(frame, policy) == Status::kUnsupportedCamera);

  std::array<std::uint16_t, 4 * 8 * 8> input{};
  std::array<std::uint16_t, 4 * 8 * 8> output{};
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<std::uint16_t>(index);
  }
  PackedRawView source{input.data(), 8, 8, 16, 128};
  MutablePackedRawView destination{output.data(), 8, 8, 16, 128};
  assert(BitExactBypass(source, destination) == Status::kOk);
  assert(input == output);

  TestAllCfaRoundTrips();
  TestDmaBufLifecycle();

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
  UnavailablePostProcessExecutor postprocessor;
  std::uint64_t fence = 0;
  assert(postprocessor.Execute(nullptr, nullptr, RyybCfaPhase::kRyyb, 40, &fence) ==
         Status::kNpuUnavailable);
  return 0;
}

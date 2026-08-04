#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ai_isp {

// 状态码与 V3.0 发布接口一致；任何失败都必须输出原始 RAW。
enum class Status {
  kOk,
  kBypassBright,
  kBypassCameraTransition,
  kInvalidArgument,
  kUnsupportedProfile,
  kInvalidMetadata,
  kSchemaMismatch,
  kHashMismatch,
  kGearSetFailed,
  kNpuUnavailable,
  kNpuTimeout,
  kOutOfMemory,
  kInferenceError,
  kNonfiniteOutput,
  kThermalBypass,
};

enum class ProfileId { kP0, kP1, kP2 };

struct Profile {
  ProfileId id;
  std::uint32_t valid_width;
  std::uint32_t valid_height;
  std::uint32_t compile_width;
  std::uint32_t compile_height;
};

struct FrameMetadata {
  float exposure_time_s = 0.0F;
  float iso = 0.0F;
  float scene_ev = 0.0F;
  float noise_level = 0.0F;
  std::optional<float> dark_score;
  bool metadata_valid = false;
  bool camera_transition = false;
  bool thermal_bypass = false;
  std::uint64_t frame_id = 0;
  std::uint64_t timestamp_ns = 0;
};

struct PackedRawView {
  // 四平面 uint16 容器，plane_stride_bytes 表示相邻平面的跨度。
  const std::uint16_t* data = nullptr;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t row_stride_bytes = 0;
  std::size_t plane_stride_bytes = 0;
};

struct MutablePackedRawView {
  std::uint16_t* data = nullptr;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t row_stride_bytes = 0;
  std::size_t plane_stride_bytes = 0;
};

struct TriggerDecision {
  bool bypass = true;
  float enhancement_strength = 0.0F;
};

const Profile* SelectProfile(std::uint32_t valid_width, std::uint32_t valid_height);
bool ValidateBuffer(const PackedRawView& input);
bool ValidateBuffer(const MutablePackedRawView& output);
Status BitExactBypass(const PackedRawView& input, const MutablePackedRawView& output);

class DarkTrigger {
 public:
  TriggerDecision Update(const FrameMetadata& metadata);
  TriggerDecision FailImmediately();
  void Recover();

 private:
  enum class State { kBypassBright, kArming, kActiveRamp, kActive, kExitPending, kBypassError };
  State state_ = State::kBypassBright;
  std::uint32_t enter_counter_ = 0;
  std::uint32_t exit_counter_ = 0;
  std::uint32_t ramp_index_ = 0;
};

class NpuExecutor {
 public:
  virtual ~NpuExecutor() = default;
  virtual Status Load(ProfileId profile) = 0;
  virtual Status Execute(const float* packed_input, const float* condition,
                         float* noise_output, const Profile& profile) = 0;
};

// 本地/未接 DDK 时使用的明确失败实现，禁止静默执行 CPU AI。
class UnavailableNpuExecutor final : public NpuExecutor {
 public:
  Status Load(ProfileId profile) override;
  Status Execute(const float* packed_input, const float* condition,
                 float* noise_output, const Profile& profile) override;
};

}  // namespace ai_isp


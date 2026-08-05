#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ai_isp {

// V4 任何失败都必须输出原始 RAW 或交回传统 ISP NR。
enum class Status {
  kOk,
  kBypassBright,
  kBypassCameraTransition,
  kInvalidArgument,
  kUnsupportedCamera,
  kInvalidCfaPhase,
  kInvalidMetadata,
  kSchemaMismatch,
  kHashMismatch,
  kNpuUnavailable,
  kNpuTimeout,
  kOutOfMemory,
  kInferenceError,
  kNonfiniteOutput,
  kThermalBypass,
};

enum class CameraId { kMain, kTele, kUltrawide, kOther };
enum class RyybCfaPhase { kRyyb, kByyr, kYryb, kYbyr };

struct Profile {
  std::string_view id;
  std::uint32_t valid_width;
  std::uint32_t valid_height;
  std::uint32_t compile_width;
  std::uint32_t compile_height;
  std::uint32_t raw_width;
  std::uint32_t raw_height;
};

struct RyybFrameDescriptor {
  CameraId camera = CameraId::kOther;
  RyybCfaPhase cfa_phase = RyybCfaPhase::kRyyb;
  std::string_view sensor_profile;
  std::uint32_t raw_width = 0;
  std::uint32_t raw_height = 0;
  std::uint32_t crop_x = 0;
  std::uint32_t crop_y = 0;
  std::uint32_t crop_width = 0;
  std::uint32_t crop_height = 0;
  std::size_t row_stride_bytes = 0;
  std::uint32_t bit_depth = 0;
  std::array<float, 4> black_level{};
  std::array<float, 4> white_level{};
  std::string_view model_hash;
  std::string_view quant_policy_hash;
};

struct AdmissionPolicy {
  std::string_view main_sensor_profile;
  std::string_view tele_sensor_profile;
  RyybCfaPhase main_cfa_phase = RyybCfaPhase::kRyyb;
  RyybCfaPhase tele_cfa_phase = RyybCfaPhase::kRyyb;
  std::string_view model_hash;
  std::string_view quant_policy_hash;
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

const Profile& GetFixedRyybProfile();
Status ValidateAiAdmission(const RyybFrameDescriptor& frame, const AdmissionPolicy& policy);
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
  virtual Status Load() = 0;
  virtual Status Execute(const float* packed_input, const float* condition,
                         float* noise_output, const Profile& profile) = 0;
};

// 本地/未接 DDK 时使用的明确失败实现，禁止静默执行 CPU AI。
class UnavailableNpuExecutor final : public NpuExecutor {
 public:
  Status Load() override;
  Status Execute(const float* packed_input, const float* condition,
                 float* noise_output, const Profile& profile) override;
};

}  // namespace ai_isp

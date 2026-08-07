#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ai_isp {

// V6.1 fail-closed contract: every failure returns the original RAW or hands
// control back to the conventional ISP NR path.
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
  kBufferContractMismatch,
  kFenceOrderError,
  kBufferBusy,
  kNpuUnavailable,
  kNpuTimeout,
  kOutOfMemory,
  kInferenceError,
  kNonfiniteOutput,
  kThermalBypass,
};

enum class CameraId { kMain, kTele, kUltrawide, kOther };
enum class RyybCfaPhase { kRyyb, kByyr, kYryb, kYbyr };
enum class RawDomainState { kLinearPostBlcLscPreDgain, kUnknown };

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
  RawDomainState raw_domain_state = RawDomainState::kUnknown;
  bool blc_applied = false;
  bool lsc_applied = false;
  std::string_view raw_domain_profile_hash;
  std::string_view lsc_profile_hash;
  std::string_view unpack_profile_hash;
  std::string_view buffer_contract_version;
  int buffer_fd = -1;
  std::uint32_t buffer_index = 0;
  std::size_t plane_offset_bytes = 0;
  int producer_fence_fd = -1;
  std::size_t extra_cpu_memcpy_bytes = 0;
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
  std::string_view raw_domain_profile_hash;
  std::string_view main_lsc_profile_hash;
  std::string_view tele_lsc_profile_hash;
  std::string_view main_unpack_profile_hash;
  std::string_view tele_unpack_profile_hash;
  std::string_view buffer_contract_version = "v1";
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
  // Four semantic uint16 planes in fixed R/Yr/Yb/B order.
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

struct PhysicalRawView {
  const std::uint16_t* data = nullptr;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t row_stride_bytes = 0;
};

struct MutablePhysicalRawView {
  std::uint16_t* data = nullptr;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t row_stride_bytes = 0;
};

struct DmaBufFrame {
  std::uint32_t buffer_index = 0;
  int fd = -1;
  std::size_t size_bytes = 0;
  std::size_t plane_offset_bytes = 0;
  std::size_t row_stride_bytes = 0;
  std::uint32_t valid_width = 0;
  std::uint32_t valid_height = 0;
  std::uint64_t producer_fence = 0;
  std::size_t cpu_memcpy_bytes = 0;
  std::uint32_t map_operations = 0;
};

struct DmaBufAudit {
  std::size_t imported_buffers = 0;
  std::uint64_t submitted_frames = 0;
  std::uint64_t timeout_recoveries = 0;
  std::size_t extra_cpu_memcpy_bytes = 0;
  std::uint32_t per_frame_map_unmap = 0;
};

struct TriggerDecision {
  bool bypass = true;
  float enhancement_strength = 0.0F;
};

const Profile& GetFixedRyybProfile();
Status ValidateAiAdmission(const RyybFrameDescriptor& frame, const AdmissionPolicy& policy);
bool ValidateBuffer(const PackedRawView& input);
bool ValidateBuffer(const MutablePackedRawView& output);
bool ValidateBuffer(const PhysicalRawView& input);
bool ValidateBuffer(const MutablePhysicalRawView& output);
Status BitExactBypass(const PackedRawView& input, const MutablePackedRawView& output);

// Verification-only scalar references. Production must use PostProcessExecutor
// so there is no per-pixel CPU implementation in the camera path.
Status ReferencePackRyyb(const PhysicalRawView& input, RyybCfaPhase phase,
                         const MutablePackedRawView& output);
Status ReferenceUnpackRyyb(const PackedRawView& input, RyybCfaPhase phase,
                           const MutablePhysicalRawView& output);

class DmaBufPoolContract {
 public:
  static constexpr std::size_t kMaxBuffers = 16;
  Status ImportOnce(std::uint32_t buffer_index, int fd, std::size_t size_bytes);
  Status Submit(const DmaBufFrame& frame);
  Status SignalConsumerReady(std::uint32_t buffer_index, std::uint64_t consumer_fence);
  Status Release(std::uint32_t buffer_index, std::uint64_t waited_fence);
  Status RecoverTimeout(std::uint32_t buffer_index);
  DmaBufAudit Audit() const;

 private:
  enum class BufferState { kUnused, kIdle, kNpuInFlight, kConsumerReady };
  struct Slot {
    BufferState state = BufferState::kUnused;
    int fd = -1;
    std::size_t size_bytes = 0;
    std::uint64_t producer_fence = 0;
    std::uint64_t consumer_fence = 0;
  };
  std::array<Slot, kMaxBuffers> slots_{};
  std::uint64_t submitted_frames_ = 0;
  std::uint64_t timeout_recoveries_ = 0;
};

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

class PostProcessExecutor {
 public:
  virtual ~PostProcessExecutor() = default;
  // Performs FP16 subtract/clamp and semantic unpack on NPU/ISP vector units.
  virtual Status Execute(const void* packed_fp16, const void* noise_fp16,
                         RyybCfaPhase phase, int output_dmabuf_fd,
                         std::uint64_t* consumer_fence) = 0;
};

// Explicit local stubs: never silently fall back to CPU/GPU AI or pixel loops.
class UnavailableNpuExecutor final : public NpuExecutor {
 public:
  Status Load() override;
  Status Execute(const float* packed_input, const float* condition,
                 float* noise_output, const Profile& profile) override;
};

class UnavailablePostProcessExecutor final : public PostProcessExecutor {
 public:
  Status Execute(const void* packed_fp16, const void* noise_fp16,
                 RyybCfaPhase phase, int output_dmabuf_fd,
                 std::uint64_t* consumer_fence) override;
};

}  // namespace ai_isp

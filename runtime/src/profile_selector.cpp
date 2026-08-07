#include "dark_preview_denoise.h"

namespace ai_isp {
namespace {
constexpr Profile kFixedRyybProfile{"RYYB_4X3", 1024, 768, 1024, 768, 2048, 1536};
}  // namespace

const Profile& GetFixedRyybProfile() { return kFixedRyybProfile; }

Status ValidateAiAdmission(const RyybFrameDescriptor& frame, const AdmissionPolicy& policy) {
  if (frame.camera != CameraId::kMain && frame.camera != CameraId::kTele) {
    return Status::kUnsupportedCamera;
  }
  const auto expected_sensor =
      frame.camera == CameraId::kMain ? policy.main_sensor_profile : policy.tele_sensor_profile;
  if (frame.sensor_profile.empty() || frame.sensor_profile != expected_sensor) {
    return Status::kSchemaMismatch;
  }
  const auto expected_cfa =
      frame.camera == CameraId::kMain ? policy.main_cfa_phase : policy.tele_cfa_phase;
  if (frame.cfa_phase != expected_cfa) {
    return Status::kInvalidCfaPhase;
  }
  if ((frame.crop_x & 1U) || (frame.crop_y & 1U) || (frame.crop_width & 1U) ||
      (frame.crop_height & 1U)) {
    return Status::kInvalidCfaPhase;
  }
  if (frame.raw_width != kFixedRyybProfile.raw_width ||
      frame.raw_height != kFixedRyybProfile.raw_height ||
      frame.crop_width != kFixedRyybProfile.raw_width ||
      frame.crop_height != kFixedRyybProfile.raw_height) {
    return Status::kInvalidArgument;
  }
  if (frame.row_stride_bytes < static_cast<std::size_t>(frame.raw_width) * sizeof(std::uint16_t) ||
      (frame.bit_depth != 10 && frame.bit_depth != 12 && frame.bit_depth != 14 &&
       frame.bit_depth != 16)) {
    return Status::kInvalidArgument;
  }
  for (std::size_t index = 0; index < frame.black_level.size(); ++index) {
    if (!(frame.white_level[index] > frame.black_level[index])) {
      return Status::kInvalidMetadata;
    }
  }
  if (frame.raw_domain_state != RawDomainState::kLinearPostBlcLscPreDgain ||
      !frame.blc_applied || !frame.lsc_applied) {
    return Status::kInvalidMetadata;
  }
  if ((!policy.raw_domain_profile_hash.empty() &&
       frame.raw_domain_profile_hash != policy.raw_domain_profile_hash) ||
      frame.buffer_contract_version != policy.buffer_contract_version) {
    return Status::kBufferContractMismatch;
  }
  const auto expected_lsc =
      frame.camera == CameraId::kMain ? policy.main_lsc_profile_hash : policy.tele_lsc_profile_hash;
  const auto expected_unpack = frame.camera == CameraId::kMain
                                   ? policy.main_unpack_profile_hash
                                   : policy.tele_unpack_profile_hash;
  if ((!expected_lsc.empty() && frame.lsc_profile_hash != expected_lsc) ||
      (!expected_unpack.empty() && frame.unpack_profile_hash != expected_unpack)) {
    return Status::kHashMismatch;
  }
  if (frame.buffer_fd < 0 || (frame.plane_offset_bytes & 1U) ||
      frame.extra_cpu_memcpy_bytes != 0U) {
    return Status::kBufferContractMismatch;
  }
  if ((!policy.model_hash.empty() && frame.model_hash != policy.model_hash) ||
      (!policy.quant_policy_hash.empty() && frame.quant_policy_hash != policy.quant_policy_hash)) {
    return Status::kHashMismatch;
  }
  return Status::kOk;
}

}  // namespace ai_isp

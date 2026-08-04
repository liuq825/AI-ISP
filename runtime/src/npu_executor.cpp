#include "dark_preview_denoise.h"

namespace ai_isp {

Status UnavailableNpuExecutor::Load(ProfileId /*profile*/) {
  return Status::kNpuUnavailable;
}

Status UnavailableNpuExecutor::Execute(const float* /*packed_input*/,
                                       const float* /*condition*/,
                                       float* /*noise_output*/,
                                       const Profile& /*profile*/) {
  // 明确失败，不允许在量产关键路径自动改用 CPU/GPU AI。
  return Status::kNpuUnavailable;
}

}  // namespace ai_isp


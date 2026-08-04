#include "dark_preview_denoise.h"

#include <array>

namespace ai_isp {
namespace {
constexpr std::array<Profile, 3> kProfiles{{
    {ProfileId::kP0, 1024, 768, 1024, 768},
    {ProfileId::kP1, 960, 540, 960, 544},
    {ProfileId::kP2, 960, 640, 960, 640},
}};
}  // namespace

const Profile* SelectProfile(const std::uint32_t valid_width,
                             const std::uint32_t valid_height) {
  for (const auto& profile : kProfiles) {
    if (profile.valid_width == valid_width && profile.valid_height == valid_height) {
      return &profile;
    }
  }
  return nullptr;
}

}  // namespace ai_isp


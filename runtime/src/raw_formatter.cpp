#include "dark_preview_denoise.h"

namespace ai_isp {

bool ValidateBuffer(const PackedRawView& input) {
  const auto minimum_row = static_cast<std::size_t>(input.width) * sizeof(std::uint16_t);
  const auto minimum_plane = input.row_stride_bytes * input.height;
  return input.data != nullptr && input.width > 0 && input.height > 0 &&
         input.row_stride_bytes >= minimum_row && input.plane_stride_bytes >= minimum_plane;
}

bool ValidateBuffer(const MutablePackedRawView& output) {
  const auto minimum_row = static_cast<std::size_t>(output.width) * sizeof(std::uint16_t);
  const auto minimum_plane = output.row_stride_bytes * output.height;
  return output.data != nullptr && output.width > 0 && output.height > 0 &&
         output.row_stride_bytes >= minimum_row && output.plane_stride_bytes >= minimum_plane;
}

}  // namespace ai_isp


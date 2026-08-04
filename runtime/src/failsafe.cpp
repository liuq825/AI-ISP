#include "dark_preview_denoise.h"

#include <cstring>

namespace ai_isp {

Status BitExactBypass(const PackedRawView& input, const MutablePackedRawView& output) {
  if (!ValidateBuffer(input) || !ValidateBuffer(output) || input.width != output.width ||
      input.height != output.height) {
    return Status::kInvalidArgument;
  }
  const auto row_bytes = static_cast<std::size_t>(input.width) * sizeof(std::uint16_t);
  const auto* input_bytes = reinterpret_cast<const std::byte*>(input.data);
  auto* output_bytes = reinterpret_cast<std::byte*>(output.data);
  for (std::size_t plane = 0; plane < 4; ++plane) {
    for (std::size_t row = 0; row < input.height; ++row) {
      std::memcpy(output_bytes + plane * output.plane_stride_bytes + row * output.row_stride_bytes,
                  input_bytes + plane * input.plane_stride_bytes + row * input.row_stride_bytes,
                  row_bytes);
    }
  }
  return Status::kOk;
}

}  // namespace ai_isp


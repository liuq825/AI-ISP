#include "dark_preview_denoise.h"

#include <array>
#include <cstddef>

namespace ai_isp {

namespace {

using Offsets = std::array<std::array<std::size_t, 2>, 4>;

constexpr Offsets kRyyb{{{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}}};
constexpr Offsets kByyr{{{{1, 1}}, {{1, 0}}, {{0, 1}}, {{0, 0}}}};
constexpr Offsets kYryb{{{{0, 1}}, {{0, 0}}, {{1, 1}}, {{1, 0}}}};
constexpr Offsets kYbyr{{{{1, 0}}, {{1, 1}}, {{0, 0}}, {{0, 1}}}};

const Offsets& GetOffsets(RyybCfaPhase phase) {
  switch (phase) {
    case RyybCfaPhase::kRyyb: return kRyyb;
    case RyybCfaPhase::kByyr: return kByyr;
    case RyybCfaPhase::kYryb: return kYryb;
    case RyybCfaPhase::kYbyr: return kYbyr;
  }
  return kRyyb;
}

const std::uint16_t* PackedRow(const PackedRawView& view, std::size_t plane, std::size_t row) {
  const auto* bytes = reinterpret_cast<const std::byte*>(view.data);
  return reinterpret_cast<const std::uint16_t*>(
      bytes + plane * view.plane_stride_bytes + row * view.row_stride_bytes);
}

std::uint16_t* PackedRow(const MutablePackedRawView& view, std::size_t plane, std::size_t row) {
  auto* bytes = reinterpret_cast<std::byte*>(view.data);
  return reinterpret_cast<std::uint16_t*>(
      bytes + plane * view.plane_stride_bytes + row * view.row_stride_bytes);
}

const std::uint16_t* PhysicalRow(const PhysicalRawView& view, std::size_t row) {
  const auto* bytes = reinterpret_cast<const std::byte*>(view.data);
  return reinterpret_cast<const std::uint16_t*>(bytes + row * view.row_stride_bytes);
}

std::uint16_t* PhysicalRow(const MutablePhysicalRawView& view, std::size_t row) {
  auto* bytes = reinterpret_cast<std::byte*>(view.data);
  return reinterpret_cast<std::uint16_t*>(bytes + row * view.row_stride_bytes);
}

}  // namespace

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

bool ValidateBuffer(const PhysicalRawView& input) {
  return input.data != nullptr && input.width > 0 && input.height > 0 &&
         input.row_stride_bytes >= static_cast<std::size_t>(input.width) * sizeof(std::uint16_t);
}

bool ValidateBuffer(const MutablePhysicalRawView& output) {
  return output.data != nullptr && output.width > 0 && output.height > 0 &&
         output.row_stride_bytes >= static_cast<std::size_t>(output.width) * sizeof(std::uint16_t);
}

Status ReferencePackRyyb(const PhysicalRawView& input, RyybCfaPhase phase,
                         const MutablePackedRawView& output) {
  if (!ValidateBuffer(input) || !ValidateBuffer(output) || (input.width & 1U) ||
      (input.height & 1U) || output.width * 2U != input.width ||
      output.height * 2U != input.height) {
    return Status::kInvalidArgument;
  }
  const auto& offsets = GetOffsets(phase);
  for (std::size_t channel = 0; channel < offsets.size(); ++channel) {
    const auto row_offset = offsets[channel][0];
    const auto column_offset = offsets[channel][1];
    for (std::size_t row = 0; row < output.height; ++row) {
      const auto* source = PhysicalRow(input, row * 2U + row_offset);
      auto* destination = PackedRow(output, channel, row);
      for (std::size_t column = 0; column < output.width; ++column) {
        destination[column] = source[column * 2U + column_offset];
      }
    }
  }
  return Status::kOk;
}

Status ReferenceUnpackRyyb(const PackedRawView& input, RyybCfaPhase phase,
                           const MutablePhysicalRawView& output) {
  if (!ValidateBuffer(input) || !ValidateBuffer(output) || output.width != input.width * 2U ||
      output.height != input.height * 2U) {
    return Status::kInvalidArgument;
  }
  const auto& offsets = GetOffsets(phase);
  for (std::size_t channel = 0; channel < offsets.size(); ++channel) {
    const auto row_offset = offsets[channel][0];
    const auto column_offset = offsets[channel][1];
    for (std::size_t row = 0; row < input.height; ++row) {
      const auto* source = PackedRow(input, channel, row);
      auto* destination = PhysicalRow(output, row * 2U + row_offset);
      for (std::size_t column = 0; column < input.width; ++column) {
        destination[column * 2U + column_offset] = source[column];
      }
    }
  }
  return Status::kOk;
}

}  // namespace ai_isp

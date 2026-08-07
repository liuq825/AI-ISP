#include "dark_preview_denoise.h"

namespace ai_isp {

Status DmaBufPoolContract::ImportOnce(std::uint32_t buffer_index, int fd,
                                     std::size_t size_bytes) {
  if (buffer_index >= slots_.size() || fd < 0 || size_bytes == 0) {
    return Status::kInvalidArgument;
  }
  auto& slot = slots_[buffer_index];
  if (slot.state != BufferState::kUnused) {
    return slot.fd == fd && slot.size_bytes == size_bytes ? Status::kOk
                                                          : Status::kBufferContractMismatch;
  }
  slot.state = BufferState::kIdle;
  slot.fd = fd;
  slot.size_bytes = size_bytes;
  return Status::kOk;
}

Status DmaBufPoolContract::Submit(const DmaBufFrame& frame) {
  if (frame.buffer_index >= slots_.size()) return Status::kInvalidArgument;
  auto& slot = slots_[frame.buffer_index];
  if (slot.state == BufferState::kUnused || slot.fd != frame.fd ||
      slot.size_bytes != frame.size_bytes) {
    return Status::kBufferContractMismatch;
  }
  if (slot.state != BufferState::kIdle) return Status::kBufferBusy;
  if (frame.valid_width == 0 || frame.valid_height == 0 ||
      frame.row_stride_bytes < static_cast<std::size_t>(frame.valid_width) * sizeof(std::uint16_t) ||
      (frame.plane_offset_bytes & 1U) || frame.cpu_memcpy_bytes != 0 ||
      frame.map_operations != 0) {
    return Status::kBufferContractMismatch;
  }
  if (frame.producer_fence <= slot.producer_fence) return Status::kFenceOrderError;
  slot.producer_fence = frame.producer_fence;
  slot.state = BufferState::kNpuInFlight;
  ++submitted_frames_;
  return Status::kOk;
}

Status DmaBufPoolContract::SignalConsumerReady(std::uint32_t buffer_index,
                                               std::uint64_t consumer_fence) {
  if (buffer_index >= slots_.size()) return Status::kInvalidArgument;
  auto& slot = slots_[buffer_index];
  if (slot.state != BufferState::kNpuInFlight) return Status::kBufferContractMismatch;
  if (consumer_fence <= slot.consumer_fence) return Status::kFenceOrderError;
  slot.consumer_fence = consumer_fence;
  slot.state = BufferState::kConsumerReady;
  return Status::kOk;
}

Status DmaBufPoolContract::Release(std::uint32_t buffer_index, std::uint64_t waited_fence) {
  if (buffer_index >= slots_.size()) return Status::kInvalidArgument;
  auto& slot = slots_[buffer_index];
  if (slot.state != BufferState::kConsumerReady) return Status::kBufferContractMismatch;
  if (waited_fence != slot.consumer_fence) return Status::kFenceOrderError;
  slot.state = BufferState::kIdle;
  return Status::kOk;
}

Status DmaBufPoolContract::RecoverTimeout(std::uint32_t buffer_index) {
  if (buffer_index >= slots_.size()) return Status::kInvalidArgument;
  auto& slot = slots_[buffer_index];
  if (slot.state != BufferState::kNpuInFlight && slot.state != BufferState::kConsumerReady) {
    return Status::kBufferContractMismatch;
  }
  slot.state = BufferState::kIdle;
  ++timeout_recoveries_;
  return Status::kOk;
}

DmaBufAudit DmaBufPoolContract::Audit() const {
  DmaBufAudit audit{};
  for (const auto& slot : slots_) {
    if (slot.state != BufferState::kUnused) ++audit.imported_buffers;
  }
  audit.submitted_frames = submitted_frames_;
  audit.timeout_recoveries = timeout_recoveries_;
  // These remain hard-coded gates because Submit rejects either operation.
  audit.extra_cpu_memcpy_bytes = 0;
  audit.per_frame_map_unmap = 0;
  return audit;
}

}  // namespace ai_isp

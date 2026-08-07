"""DMA-BUF 池化共享、Fence 与零 CPU 拷贝的可执行契约模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BufferState(str, Enum):
    IDLE = "idle"
    NPU_IN_FLIGHT = "npu_in_flight"
    CONSUMER_READY = "consumer_ready"


@dataclass(frozen=True)
class DmaBufFrame:
    buffer_index: int
    fd: int
    plane_offset_bytes: int
    row_stride_bytes: int
    valid_width: int
    valid_height: int
    producer_fence: int
    cpu_memcpy_bytes: int = 0
    map_operations: int = 0


@dataclass
class _ImportedBuffer:
    fd: int
    size_bytes: int
    state: BufferState = BufferState.IDLE
    last_producer_fence: int = -1
    last_consumer_fence: int = -1


class DmaBufPoolContract:
    """不执行真实 DMA；用于在 HAL/NPU 接口接入前验证生命周期不变量。"""

    def __init__(self, version: str = "v1") -> None:
        if version != "v1":
            raise ValueError("当前只支持 buffer_contract_version=v1")
        self.version = version
        self._buffers: dict[int, _ImportedBuffer] = {}
        self.import_count = 0
        self.frame_count = 0
        self.timeout_recovery_count = 0

    def import_once(self, buffer_index: int, fd: int, size_bytes: int) -> None:
        if buffer_index < 0 or fd < 0 or size_bytes <= 0:
            raise ValueError("DMA-BUF 导入参数非法")
        existing = self._buffers.get(buffer_index)
        if existing is not None:
            if existing.fd != fd or existing.size_bytes != size_bytes:
                raise ValueError("同一 Buffer Index 禁止逐帧更换 FD 或尺寸")
            return
        self._buffers[buffer_index] = _ImportedBuffer(fd=fd, size_bytes=size_bytes)
        self.import_count += 1

    def submit(self, frame: DmaBufFrame) -> None:
        buffer = self._buffers.get(frame.buffer_index)
        if buffer is None or buffer.fd != frame.fd:
            raise ValueError("Buffer 必须在 Stream 初始化时预先导入")
        if buffer.state is not BufferState.IDLE:
            raise RuntimeError(f"Buffer {frame.buffer_index} 尚未回到 IDLE")
        if frame.plane_offset_bytes < 0 or frame.plane_offset_bytes % 2:
            raise ValueError("Plane Offset 必须按 uint16 对齐")
        if frame.row_stride_bytes < frame.valid_width * 2:
            raise ValueError("Row Stride 小于有效 uint16 RAW 行宽")
        if frame.valid_width <= 0 or frame.valid_height <= 0:
            raise ValueError("有效区宽高必须为正数")
        if frame.producer_fence <= buffer.last_producer_fence:
            raise ValueError("Producer Fence 必须单调递增")
        if frame.cpu_memcpy_bytes != 0 or frame.map_operations != 0:
            raise ValueError("每帧额外 CPU memcpy 与 map/unmap 必须为 0")
        buffer.last_producer_fence = frame.producer_fence
        buffer.state = BufferState.NPU_IN_FLIGHT
        self.frame_count += 1

    def signal_consumer_ready(self, buffer_index: int, consumer_fence: int) -> None:
        buffer = self._require(buffer_index)
        if buffer.state is not BufferState.NPU_IN_FLIGHT:
            raise RuntimeError("只有 NPU_IN_FLIGHT Buffer 可发布 Consumer Fence")
        if consumer_fence <= buffer.last_consumer_fence:
            raise ValueError("Consumer Fence 必须单调递增")
        buffer.last_consumer_fence = consumer_fence
        buffer.state = BufferState.CONSUMER_READY

    def release(self, buffer_index: int, waited_fence: int) -> None:
        buffer = self._require(buffer_index)
        if buffer.state is not BufferState.CONSUMER_READY:
            raise RuntimeError("下游尚未获得 Buffer")
        if waited_fence != buffer.last_consumer_fence:
            raise ValueError("释放前必须等待当前 Consumer Fence")
        buffer.state = BufferState.IDLE

    def recover_timeout(self, buffer_index: int) -> None:
        buffer = self._require(buffer_index)
        if buffer.state is BufferState.IDLE:
            raise RuntimeError("IDLE Buffer 不需要超时回收")
        buffer.state = BufferState.IDLE
        self.timeout_recovery_count += 1

    def _require(self, buffer_index: int) -> _ImportedBuffer:
        if buffer_index not in self._buffers:
            raise ValueError("未知 Buffer Index")
        return self._buffers[buffer_index]

    def audit(self) -> dict[str, object]:
        return {
            "buffer_contract_version": self.version,
            "buffer_count": len(self._buffers),
            "fd_import_count": self.import_count,
            "frame_count": self.frame_count,
            "timeout_recovery_count": self.timeout_recovery_count,
            "extra_cpu_memcpy_bytes": 0,
            "per_frame_map_unmap": 0,
            "states": {index: item.state.value for index, item in sorted(self._buffers.items())},
        }

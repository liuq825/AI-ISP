import pytest

from ai_isp.runtime import DmaBufFrame, DmaBufPoolContract


def _frame(fence: int = 1, memcpy: int = 0) -> DmaBufFrame:
    return DmaBufFrame(0, 10, 0, 4096, 2048, 1536, fence, cpu_memcpy_bytes=memcpy)


def test_dmabuf_fd_is_imported_once_and_fences_control_reuse() -> None:
    pool = DmaBufPoolContract()
    pool.import_once(0, 10, 2048 * 1536 * 2)
    pool.import_once(0, 10, 2048 * 1536 * 2)
    pool.submit(_frame())
    pool.signal_consumer_ready(0, 2)
    pool.release(0, 2)
    audit = pool.audit()
    assert audit["fd_import_count"] == 1
    assert audit["extra_cpu_memcpy_bytes"] == 0
    assert audit["states"] == {0: "idle"}


def test_dmabuf_contract_rejects_memcpy_and_recovers_timeout() -> None:
    pool = DmaBufPoolContract()
    pool.import_once(0, 10, 4096)
    with pytest.raises(ValueError, match="memcpy"):
        pool.submit(_frame(memcpy=4))
    pool.submit(_frame())
    pool.recover_timeout(0)
    assert pool.audit()["timeout_recovery_count"] == 1

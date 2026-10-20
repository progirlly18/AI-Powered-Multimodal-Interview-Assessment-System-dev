"""Lazy model registry."""

import threading

import pytest

from backend.core.models import ModelLoadError, ModelRegistry, ModelSpec, build_default_registry
from backend.core.states import DeviceStatus


def test_loads_once_and_records_time():
    calls = []
    reg = ModelRegistry({"m": ModelSpec("Model", lambda: calls.append(1) or object(), "help")})
    assert reg.info("m").status is DeviceStatus.NOT_CHECKED
    first = reg.get("m")
    assert reg.get("m") is first and len(calls) == 1
    assert reg.info("m").status is DeviceStatus.READY and reg.info("m").load_seconds >= 0


def test_concurrent_get_loads_once():
    calls = []
    barrier = threading.Barrier(5)

    def loader():
        calls.append(1)
        return object()

    reg = ModelRegistry({"m": ModelSpec("Model", loader, "")})

    def worker():
        barrier.wait()
        reg.get("m")

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def test_failure_is_reported_and_not_cached():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("network down")
        return "model"

    reg = ModelRegistry({"m": ModelSpec("Speech model", flaky, "Check your connection.")})
    with pytest.raises(ModelLoadError, match="Speech model could not be loaded. Check your connection."):
        reg.get("m")
    assert reg.info("m").status is DeviceStatus.FAILED and "network down" in reg.info("m").error
    assert reg.get("m") == "model"  # retry works
    assert reg.info("m").status is DeviceStatus.READY and reg.info("m").error is None


def test_missing_package_is_not_blamed_on_the_network():
    """Regression: a missing pkg_resources was reported as 'check your connection'."""
    def loader():
        raise ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources")
    reg = ModelRegistry({"m": ModelSpec("YAMNet", loader, "check your connection")})
    with pytest.raises(ModelLoadError) as info:
        reg.get("m")
    assert "pkg_resources" in str(info.value) and "requirements-lock.txt" in str(info.value)
    assert "connection" not in str(info.value)


def test_unknown_model():
    with pytest.raises(KeyError):
        ModelRegistry({}).get("x")


def test_missing_model_files_fail_cleanly(tmp_path):
    from backend.config import Settings

    reg = build_default_registry(Settings(emotion_model_path=tmp_path / "missing.keras",
                                          face_landmarker_path=tmp_path / "missing.task"))
    for name in ("emotion", "face_mesh"):
        with pytest.raises(ModelLoadError, match="not found: .*missing"):
            reg.get(name)
        assert "not found" in reg.info(name).error

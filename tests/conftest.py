import pytest


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    """Point the spinlabs registry at a per-test directory."""
    monkeypatch.setenv("SPINLABS_REGISTRY_DIR", str(tmp_path / ".spinlabs"))
    import spinlabs.registry as reg

    monkeypatch.setattr(reg, "REGISTRY_DIR", tmp_path / ".spinlabs")
    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / ".spinlabs" / "registry.db")
    yield

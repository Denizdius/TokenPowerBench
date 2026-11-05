"""Runtime compatibility shims for installed third-party packages."""

from __future__ import annotations

_INSTALLED = False


def allow_huggingface_hub_v2() -> None:
    """Let Transformers import when huggingface-hub 2.x is installed.

    Transformers runs ``require_version("huggingface-hub>=1.5.0,<2.0")`` while
    it is being imported. The finder below loads that check, but ignores the
    hub pin so the installed 2.x client can stay in place.
    """
    global _INSTALLED
    import sys

    if not _INSTALLED:
        sys.meta_path.insert(0, _HubPinFinder())
        _INSTALLED = True
    _silence_loaded_hub_pin()


def _silence_loaded_hub_pin() -> None:
    import sys

    versions = sys.modules.get("transformers.utils.versions")
    if versions is None or getattr(versions, "_tokenpowerbench_hub_v2", False):
        return
    original = versions.require_version

    def require_version(requirement, hint=None):
        if isinstance(requirement, str) and requirement.startswith("huggingface-hub"):
            return None
        return original(requirement, hint)

    versions.require_version = require_version
    versions._tokenpowerbench_hub_v2 = True


class _HubPinFinder:
    """Run Transformers' dependency check with the hub 2.x pin ignored."""

    def __init__(self) -> None:
        self._busy = False

    def find_spec(self, fullname, path, target=None):
        if fullname != "transformers.dependency_versions_check" or self._busy:
            return None
        import importlib.util
        import sys

        self._busy = True
        try:
            removed = False
            if self in sys.meta_path:
                sys.meta_path.remove(self)
                removed = True
            spec = importlib.util.find_spec(fullname)
        finally:
            if removed and self not in sys.meta_path:
                sys.meta_path.insert(0, self)
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _HubPinLoader(spec.loader)
        return spec


class _HubPinLoader:
    def __init__(self, loader) -> None:
        self._loader = loader

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module) -> None:
        import importlib

        versions = importlib.import_module("transformers.utils.versions")
        _silence_module(versions)
        self._loader.exec_module(module)


def _silence_module(versions) -> None:
    if getattr(versions, "_tokenpowerbench_hub_v2", False):
        return
    original = versions.require_version

    def require_version(requirement, hint=None):
        if isinstance(requirement, str) and requirement.startswith("huggingface-hub"):
            return None
        return original(requirement, hint)

    versions.require_version = require_version
    versions._tokenpowerbench_hub_v2 = True

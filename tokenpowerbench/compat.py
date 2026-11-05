"""Runtime compatibility shims for installed third-party packages."""

from __future__ import annotations


def allow_huggingface_hub_v2() -> None:
    """Skip Transformers' ``huggingface-hub<2`` runtime pin.

    Transformers raises on import when hub 2.x is installed. The pin is a
    version check, not a missing API. Installing a no-op check module before
    the first Transformers import lets the benchmark keep the current hub.
    """
    import sys
    import types

    if sys.modules.get("transformers") not in (None,):
        return
    name = "transformers.dependency_versions_check"
    if name in sys.modules and sys.modules[name] is not None:
        return
    module = types.ModuleType(name)
    module.deps = {}
    module.pkgs_to_check_at_runtime = []
    sys.modules[name] = module

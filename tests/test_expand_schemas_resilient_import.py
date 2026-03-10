from __future__ import annotations

import builtins
import importlib
import sys

import numpy as np
import pytest


def test_expand_schemas_import_without_skimage_keeps_stft_registered(monkeypatch):
    for mod_name in list(sys.modules):
        if mod_name == "skimage" or mod_name.startswith("skimage."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if str(name).startswith("skimage"):
            raise ImportError("simulated missing skimage")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    module = importlib.import_module("src.tools.expand_schemas")
    importlib.reload(module)

    from src.tools.signal_processing_schemas import get_operator

    stft_cls = get_operator("stft")
    assert stft_cls is not None

    patch_cls = get_operator("patch")
    with pytest.raises(ImportError):
        patch_cls(patch_size=16, stride=4).execute(np.zeros((1, 64, 1), dtype=np.float32))

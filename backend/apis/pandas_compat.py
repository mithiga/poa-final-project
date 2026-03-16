"""Pandas compatibility helpers used across training and inference code paths."""

from __future__ import annotations

import inspect

import pandas as pd


def patch_stringdtype_unpickle_compat() -> None:
    """Allow loading pickles created with pandas StringDtype(storage, na_value).

    Pandas 2.3 added the ``na_value`` parameter to ``StringDtype.__init__``.
    Artifacts serialized in that environment can fail to unpickle on older
    runtimes unless we accept and ignore the extra argument.
    """
    try:
        sig = inspect.signature(pd.StringDtype.__init__)
        if "na_value" in sig.parameters:
            return

        original_init = pd.StringDtype.__init__

        def _compat_init(self, storage=None, na_value=None):
            original_init(self, storage=storage)

        pd.StringDtype.__init__ = _compat_init
    except Exception:
        # Best-effort patch only; leave default behavior if introspection fails.
        return

"""API package initialization."""

from .pandas_compat import patch_stringdtype_unpickle_compat

patch_stringdtype_unpickle_compat()


"""Fixture that fails at import time: the checker must record a skip."""

raise RuntimeError("boom at import")

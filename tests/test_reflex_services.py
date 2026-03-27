import importlib

import pytest


def test_reflex_module_is_not_available_anymore():
    """The project explicitly removed the Reflex interface in v4.8.0."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("automlops_reflex")
import inspect

# Mock a class that behaves like the problematic RunInfo
class MockRunInfo:
    def __init__(self, run_uuid, run_id, experiment_id):
        self.run_uuid = run_uuid
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.initialized = True

# The Patch Logic
def apply_patch(cls):
    _orig_init = cls.__init__
    _sig = inspect.signature(_orig_init)
    
    # We want to patch if run_uuid is mandatory (no default)
    if "run_uuid" in _sig.parameters and _sig.parameters["run_uuid"].default is inspect.Parameter.empty:
        def _patched_init(self, *args, **kwargs):
            if "run_uuid" not in kwargs and len(args) == 0:
                kwargs["run_uuid"] = kwargs.get("run_id")
            return _orig_init(self, *args, **kwargs)
        cls.__init__ = _patched_init


def test_run_info_patch_fills_run_uuid_from_run_id():
    # Precondition: original signature requires run_uuid
    try:
        MockRunInfo(run_id="test_id", experiment_id="0")
        assert False, "Expected TypeError before patch"
    except TypeError:
        pass

    apply_patch(MockRunInfo)
    obj = MockRunInfo(run_id="test_id", experiment_id="0")
    assert obj.run_uuid == "test_id"
    assert obj.run_id == "test_id"
    assert obj.experiment_id == "0"

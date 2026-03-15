import inspect

# Mock a class that behaves like the problematic RunInfo
class MockRunInfo:
    def __init__(self, run_uuid, run_id, experiment_id):
        self.run_uuid = run_uuid
        self.run_id = run_id
        self.experiment_id = experiment_id
        print(f"MockRunInfo initialized with run_uuid={run_uuid}, run_id={run_id}")

# The Patch Logic
def apply_patch(cls):
    _orig_init = cls.__init__
    _sig = inspect.signature(_orig_init)
    
    # We want to patch if run_uuid is mandatory (no default)
    if "run_uuid" in _sig.parameters and _sig.parameters["run_uuid"].default is inspect.Parameter.empty:
        print("Patching...")
        def _patched_init(self, *args, **kwargs):
            if "run_uuid" not in kwargs and len(args) == 0:
                print("Patch: run_uuid missing, using run_id")
                kwargs["run_uuid"] = kwargs.get("run_id")
            return _orig_init(self, *args, **kwargs)
        cls.__init__ = _patched_init

# Verify the issue before patch
print("Before patch:")
try:
    MockRunInfo(run_id="test_id", experiment_id="0")
except TypeError as e:
    print(f"Expected failure: {e}")

# Apply patch
apply_patch(MockRunInfo)

# Verify the fix
print("\nAfter patch:")
try:
    obj = MockRunInfo(run_id="test_id", experiment_id="0")
    print("Success after patch!")
except Exception as e:
    print(f"Failed after patch: {e}")

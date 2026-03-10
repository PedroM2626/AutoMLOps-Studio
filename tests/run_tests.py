import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    print("Running AutoMLOps Studio Test Suite...")
    # Discover and run all tests in the tests/ directory
    retcode = pytest.main(["tests", "-v"])
    
    if retcode == 0:
        print("\nALL TESTS PASSED")
        with open("test_status.txt", "w") as f:
            f.write("TESTS COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(retcode)

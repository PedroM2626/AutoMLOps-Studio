import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_automl_tab import TestAutoMLTab

if __name__ == '__main__':
    print("Running tests from wrapper...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAutoMLTab)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if result.wasSuccessful():
        print("ALL TESTS PASSED")
        with open("test_status.txt", "w") as f:
            f.write("TESTS COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("TESTS FAILED")
        sys.exit(1)

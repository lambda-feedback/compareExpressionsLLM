import unittest

try:
    from .evaluation import Params, evaluation_function
except ImportError:
    from evaluation import Params, evaluation_function
from evaluation_test_cases import test_cases


class TestEvaluationFunction(unittest.TestCase):
    """
    TestCase Class used to test the algorithm.
    """

    def test_multiple_cases(self):
        passed = 0
        failed = 0

        for i, (response, answer, params, expected) in enumerate(test_cases, 1):
            with self.subTest(test_case=i):
                result = evaluation_function(response, answer, params)
                is_correct = result.get("is_correct")

                try:
                    self.assertEqual(is_correct, expected)
                    print(f"Test {i} Passed")
                    passed += 1
                except AssertionError:
                    print(f"Test {i} Failed: expected {expected}, got {is_correct}")
                    failed += 1

                    # mismatch_info があれば表示
                    mismatch_info = result.get("mismatch_info")
                    if mismatch_info:
                        print(f"Mismatch Info (Test {i}):\n{mismatch_info}")

        print(f"\n--- Summary ---\nPassed: {passed}, Failed: {failed}, Total: {passed + failed}")


if __name__ == "__main__":
    unittest.main()

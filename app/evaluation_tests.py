import unittest

try:
    from .evaluation import Params, evaluation_function
except ImportError:
    from evaluation import Params, evaluation_function
from evaluation_test_cases import test_cases, test_cases2,test_cases3


class TestEvaluationFunction(unittest.TestCase):
    """
    TestCase Class used to test the algorithm.
    """

    def test_multiple_cases(self):
        passed = 0
        failed = 0
        case = [test_cases, test_cases2, test_cases3]
        for i, (response, answer, params, expected) in enumerate(case[2], 1): #change here test_cases <-> test_cases2
            with self.subTest(test_case=i):
                result = evaluation_function(response, answer, params)
                is_correct = result.get("is_correct")

                try:
                    self.assertEqual(is_correct, expected)
                    print(f"Test {i} Passed")
                    passed += 1
                except AssertionError:
                    print(f"Test {i} Failed:")
                    print(f"  Response: {response}")
                    print(f"  Answer  : {answer}")
                    print(f"  Params  : {params}")
                    print(f"  Expected: {expected}, Got: {is_correct}")
                    failed += 1

        print(f"\n--- Summary ---\nPassed: {passed}, Failed: {failed}, Total: {passed + failed}")


if __name__ == "__main__":
    unittest.main()
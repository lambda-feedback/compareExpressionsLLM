import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from preview import preview_function

from preview_test_cases import preview_test_cases


class TestPreviewFunction(unittest.TestCase):
    """
    TestCase Class used to test the preview_function.
    """

    def test_multiple_cases(self):
        passed = 0
        failed = 0

        for i, (response, expected_answer, expected_correct) in enumerate(preview_test_cases, 1):
            with self.subTest(test_case=i):
                result = preview_function(response, {})  # mapping is always {}
                is_correct = (result.replace(" ", "") == expected_answer.replace(" ", ""))

                try:
                    self.assertEqual(is_correct, expected_correct)
                    print(f"Test {i} Passed")
                    passed += 1
                except AssertionError:
                    print(f"Test {i} Failed:")
                    print(f"  Response: {response}")
                    print(f"  Expected: {expected_correct}, Got: {is_correct}")
                    print(f"  Converted     : {result}")
                    print(f"  Expected Answer: {expected_answer}")
                    failed += 1

        print(f"\n--- Summary ---\nPassed: {passed}, Failed: {failed}, Total: {passed + failed}")


if __name__ == "__main__":
    unittest.main()

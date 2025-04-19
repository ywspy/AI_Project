import unittest
from evaluate import score


# 假设这是一个简单的程序，用于计算平方
def dummy_program(x):
    return x * x


class TestScoreFunction(unittest.TestCase):

    def setUp(self):
        """在每个测试用例前准备工作，这里我们提供测试输入数据"""
        self.test_inputs = [1, 2, 3, 4, 5]  # 你可以根据实际需要修改输入数据

    def test_performance(self):
        """测试性能评分是否正确计算"""
        result = score(dummy_program, self.test_inputs)
        self.assertGreater(result["performance"], 0, "Performance should be greater than 0.")

    def test_runtime(self):
        """测试运行时间是否在合理范围"""
        result = score(dummy_program, self.test_inputs)
        self.assertGreater(result["runtime"], 0, "Runtime should be greater than 0.")
        self.assertLess(result["runtime"], 1, "Runtime should be reasonable (less than 1 second).")

    def test_cyclomatic_complexity(self):
        """测试圈复杂度计算是否正确"""
        result = score(dummy_program, self.test_inputs)
        self.assertGreater(result["cc"], 0, "Cyclomatic complexity should be greater than 0.")

    def test_composite_score(self):
        """测试综合得分的计算"""
        result = score(dummy_program, self.test_inputs)
        self.assertGreater(result["composite"], 0, "Composite score should be greater than 0.")

    def test_invalid_program(self):
        """测试如果传入一个无效的程序，会引发错误"""

        def invalid_program(x):
            return "string"  # 无效的返回值，期望是 int 或 float

        with self.assertRaises(ValueError, msg="program must return int or float performance."):
            score(invalid_program, self.test_inputs)


if __name__ == '__main__':
    unittest.main()

import unittest

import bench_models as bm


class MathJudgeTest(unittest.TestCase):
    def test_accepts_correct_answer(self):
        self.assertTrue(bm.judge_math("20")[0])

    def test_accepts_answer_with_suffix(self):
        self.assertTrue(bm.judge_math("20%")[0])

    def test_rejects_wrong_answer(self):
        self.assertFalse(bm.judge_math("17")[0])

    def test_rejects_empty(self):
        self.assertFalse(bm.judge_math("")[0])


class CodeJudgeTest(unittest.TestCase):
    GOOD = (
        "def top_k(nums, k):\n"
        "    if not nums or k < 1 or k > len(nums):\n"
        "        return None\n"
        "    return sorted(nums, reverse=True)[k-1]\n"
    )

    def test_correct_implementation_passes(self):
        ok, detail = bm.judge_code(self.GOOD)
        self.assertTrue(ok, detail)

    def test_missing_function_fails(self):
        ok, detail = bm.judge_code("def other(nums, k):\n    return 1\n")
        self.assertFalse(ok)
        self.assertIn("top_k", detail)

    def test_raising_on_empty_list_fails(self):
        code = "def top_k(nums, k):\n    return sorted(nums, reverse=True)[k-1]\n"
        ok, detail = bm.judge_code(code)
        self.assertFalse(ok)
        self.assertIn("4/5", detail)

    def test_syntax_error_fails_cleanly(self):
        ok, detail = bm.judge_code("def top_k(nums, k)\n    return 1\n")
        self.assertFalse(ok)
        self.assertTrue(detail)

    def test_infinite_loop_is_stopped(self):
        code = "def top_k(nums, k):\n    while True:\n        pass\n"
        ok, detail = bm.judge_code(code, timeout=3)
        self.assertFalse(ok)
        self.assertIn("超时", detail)


class InstructionJudgeTest(unittest.TestCase):
    def test_valid_output_passes(self):
        ok, detail = bm.judge_instr("一丝不苟\n做事认真细致")
        self.assertTrue(ok, detail)

    def test_forbidden_char_fails(self):
        ok, detail = bm.judge_instr("一丝不苟\n认真做事的样")
        self.assertFalse(ok)
        self.assertIn("的", detail)

    def test_wrong_line_count_fails(self):
        self.assertFalse(bm.judge_instr("一丝不苟")[0])
        self.assertFalse(bm.judge_instr("一丝不苟\n做事认真\n多余一行")[0])

    def test_first_line_length_checked(self):
        self.assertFalse(bm.judge_instr("一丝不苟啊\n做事认真")[0])

    def test_second_line_length_checked(self):
        self.assertFalse(bm.judge_instr("一丝不苟\n这件事必须认真细致处理")[0])


class NeedleJudgeTest(unittest.TestCase):
    def test_needle_found(self):
        ok, detail = bm.judge_needle("暗号是紫罗兰-7391")
        self.assertTrue(ok, detail)

    def test_needle_missing(self):
        ok, detail = bm.judge_needle("紫罗兰")
        self.assertFalse(ok)
        self.assertIn("没找回", detail)


class ErrorMessageTest(unittest.TestCase):
    def test_html_page_is_compacted(self):
        msg = bm.brief(524, "<!DOCTYPE html><html><body>timeout</body></html>")
        self.assertNotIn("<", msg)
        self.assertIn("524", msg)

    def test_json_body_is_kept(self):
        self.assertIn('"error"', bm.brief(401, '{"error":"bad token"}'))

    def test_plain_text_is_kept(self):
        self.assertIn("internal", bm.brief(500, "internal error"))


class RowScoreTest(unittest.TestCase):
    def _row(self, flags, interrupted=False):
        row = bm.Row(model="m", interrupted=interrupted)
        row.math = bm.Probe(flags[0], "", 1)
        row.code = bm.Probe(flags[1], "", 1)
        row.instr = bm.Probe(flags[2], "", 1)
        row.needle = bm.Probe(flags[3], "", 1)
        return row

    def test_full_score(self):
        row = self._row([True, True, True, True])
        self.assertEqual(row.score, 4)
        self.assertEqual(row.percent, 100.0)
        self.assertEqual(row.verdict, "优秀")

    def test_partial_score(self):
        row = self._row([True, True, True, False])
        self.assertEqual(row.score, 3)
        self.assertEqual(row.percent, 75.0)
        self.assertEqual(row.verdict, "良好")

    def test_interrupted_row(self):
        row = self._row([True, True, False, False], interrupted=True)
        self.assertEqual(row.verdict, "中断")

    def test_billing_and_speed_not_scored(self):
        row = self._row([True, True, True, True])
        row.billing = bm.Probe(False, "", 1)
        row.speed = bm.Probe(False, "", 1)
        self.assertEqual(row.score, 4)


class LongTextTest(unittest.TestCase):
    def test_contains_needle_once(self):
        self.assertEqual(bm.LONG_TEXT.count(bm.NEEDLE), 1)

    def test_is_long_enough_for_truncation_test(self):
        self.assertGreater(len(bm.LONG_TEXT), 5000)


if __name__ == "__main__":
    unittest.main()

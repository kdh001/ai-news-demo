"""원본 코드 경로와 안전한 출력에 대한 오프라인 회귀 검사."""
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from news_core import Item, extract_markdown, inject_block_ids, normalize_hackernews


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "sample.json").read_text(encoding="utf-8"))
        self.start = datetime.fromisoformat("2026-01-01T00:00:00+09:00").timestamp()
        self.end = datetime.fromisoformat("2026-01-03T00:00:00+09:00").timestamp()

    def test_original_normalization_filter_and_schema(self):
        items = normalize_hackernews(self.data, self.start, self.end)
        self.assertEqual([item.title for item in items], ["AI 가상 모델 발표", "Machine Learning 가상 연구"])
        self.assertIsInstance(items[0], Item)
        self.assertEqual(items[0].as_dict()["source"], "hackernews")
        self.assertEqual(items[0].points, 15)

    def test_time_boundary_and_missing_timestamp(self):
        items = {"hits": [{"created_at_i": self.start, "title": "AI first"}, {"created_at_i": self.end, "title": "AI end"}, {"title": "AI unknown"}]}
        self.assertEqual([x.title for x in normalize_hackernews(items, self.start, self.end)], ["AI first"])

    def test_duplicates_by_url_or_id_and_keep_top(self):
        rows = [
            {"title": "AI first", "created_at_i": self.start, "url": "HTTPS://EXAMPLE.INVALID/a/", "objectID": "A"},
            {"title": "AI duplicate url", "created_at_i": self.start, "url": "https://example.invalid/a", "objectID": "B"},
            {"title": "AI duplicate id", "created_at_i": self.start, "objectID": "A"},
            {"title": "AI second", "created_at_i": self.start},
            {"title": "AI second", "created_at_i": self.start},
        ]
        items = normalize_hackernews({"hits": rows}, self.start, self.end)
        self.assertEqual([x.title for x in items], ["AI first", "AI second", "AI second"])
        self.assertEqual(normalize_hackernews({"hits": rows}, self.start, self.end, 0), [])
        self.assertEqual(len(normalize_hackernews({"hits": rows}, self.start, self.end, 1)), 1)

    def test_malformed_and_missing_values(self):
        for hit in (None, [], 3, {"title": [], "created_at_i": self.start},
                    {"title": "AI\ninjected", "created_at_i": self.start},
                    {"title": "AI ```", "created_at_i": self.start},
                    {"title": "AI\u2028injected", "created_at_i": self.start},
                    {"title": "AI", "created_at_i": self.start, "url": 0},
                    {"title": "AI", "created_at_i": self.start, "points": True}):
            with self.subTest(hit=hit), self.assertRaises(ValueError):
                normalize_hackernews({"hits": [hit]}, self.start, self.end)
        row = {"title": "AI example", "created_at_i": self.start, "points": None, "num_comments": None}
        self.assertEqual(normalize_hackernews({"hits": [row]}, self.start, self.end)[0].points, 0)
        self.assertEqual(normalize_hackernews({"hits": [{"title": "AI", "created_at_i": True}]}, self.start, self.end), [])
        with self.assertRaises(ValueError):
            normalize_hackernews({"hits": []}, self.start, self.end, -1)

    def test_invalid_cli_has_no_traceback_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bad.json"
            source.write_text('{"hits": [null]}', encoding="utf-8")
            output = Path(tmp) / "out.md"
            command = [sys.executable, "-B", str(ROOT / "ai_news.py"), "--input", str(source), "--from", "2026-01-01", "--to", "2026-01-02", "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())

    def test_original_extract_and_block_ids(self):
        markdown = extract_markdown("prefix\n```markdown\n## 연구\n### AI 항목\n> 💡 왜 중요한가: agent\n```")
        result, sections, items = inject_block_ids(markdown)
        self.assertIn("^sec-01", result)
        self.assertIn("^item-001", result)
        self.assertEqual(sections[0][0], "연구")
        self.assertGreater(items[0][3], 0)

    def test_block_ids_are_idempotent_and_preserve_existing(self):
        text = "## 기존 섹션\n\n^sec-01\n### 기존 항목\n^item-001\n본문\n## 새 섹션\n### 새 항목\n본문\n"
        first, sections, items = inject_block_ids(text)
        second, second_sections, second_items = inject_block_ids(first)
        self.assertEqual(first, second)
        self.assertEqual(sections, second_sections)
        self.assertEqual(items, second_items)
        self.assertEqual(first.count("^sec-01"), 1)
        self.assertEqual(first.count("^item-001"), 1)
        self.assertEqual(sections[1][1], "sec-02")
        self.assertEqual(items[1][1], "item-002")

    def test_block_ids_reserve_later_ids_and_keep_duplicates(self):
        text = "## 새 제목\n### 새 항목\n## 기존 제목\n^sec-01\n### 기존 항목\n^item-001\n본문\n^item-001\n"
        output, sections, items = inject_block_ids(text)
        self.assertEqual(sections[0][1], "sec-02")
        self.assertEqual(items[0][1], "item-002")
        self.assertEqual(output.count("^item-001"), 2)
        self.assertEqual(inject_block_ids(output)[0], output)

    def test_original_keyword_substring_heuristic_is_not_classifier(self):
        rows = {"hits": [{"created_at_i": self.start, "title": "chair design"}]}
        self.assertEqual(len(normalize_hackernews(rows, self.start, self.end)), 1)

    def test_cli_and_existing_file_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "digest.md"
            meta = Path(tmp) / "digest.meta.md"
            meta.write_text("보존", encoding="utf-8")
            command = [sys.executable, "-B", str(ROOT / "ai_news.py"), "--from", "2026-01-01", "--to", "2026-01-02", "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()
            self.assertIn("Machine Learning 가상 연구", original.decode())
            self.assertNotIn("AI 가상 도구", original.decode())
            self.assertNotIn("가상 스포츠 경기", original.decode())
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(meta.read_text(encoding="utf-8"), "보존")

    def test_invalid_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.md"
            command = [sys.executable, "-B", str(ROOT / "ai_news.py"), "--from", "2026-01-02", "--to", "2026-01-01", "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

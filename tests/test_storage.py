import json
import tempfile
import unittest
from pathlib import Path

from services.exceptions import StorageError
from services.storage_service import JsonStorageService


class JsonStorageServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.document_path = root / "documents.json"
        self.qa_path = root / "qa.json"
        self.feedback_path = root / "feedback.json"
        self.storage = JsonStorageService(self.document_path, self.qa_path, self.feedback_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_document_lifecycle(self):
        record = self.storage.add_document({"title": "制度", "file_hash": "abc"})
        self.assertEqual(self.storage.get_document_by_hash("abc")["id"], record["id"])
        self.storage.update_document(record["id"], {"status": "success"})
        self.assertEqual(self.storage.get_document_by_id(record["id"])["status"], "success")
        self.assertTrue(self.storage.delete_document(record["id"]))
        self.assertEqual(self.storage.list_documents(), [])

    def test_corrupted_json_is_not_silently_overwritten(self):
        self.document_path.write_text("not-json", encoding="utf-8")
        with self.assertRaises(StorageError):
            self.storage.add_document({"title": "new"})
        self.assertEqual(self.document_path.read_text(encoding="utf-8"), "not-json")

    def test_feedback_stats(self):
        self.storage.upsert_feedback("qa-1", "helpful", "", "session")
        self.storage.upsert_feedback("qa-2", "needs_improvement", "", "session")
        self.assertEqual(self.storage.get_stats()["helpful_rate"], 0.5)
        json.loads(self.feedback_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4

import config_data as config
from services.exceptions import StorageError


class JsonStorageService:
    """Atomic JSON repository suitable for a single-process demo deployment."""

    _lock = RLock()

    def __init__(self, document_path=None, qa_path=None, feedback_path=None):
        if document_path is None and qa_path is None and feedback_path is None:
            config.ensure_runtime_dirs()
        self.document_path = Path(document_path or config.DOCUMENT_INDEX_PATH)
        self.qa_path = Path(qa_path or config.QA_LOG_PATH)
        self.feedback_path = Path(feedback_path or config.FEEDBACK_PATH)
        for path in (self.document_path, self.qa_path, self.feedback_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("[]", encoding="utf-8")

    def _read_records(self, path):
        with self._lock:
            try:
                records = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return []
            except json.JSONDecodeError as exc:
                raise StorageError(f"存储文件已损坏，拒绝覆盖：{path}") from exc
            if not isinstance(records, list):
                raise StorageError(f"存储文件格式错误，应为 JSON 数组：{path}")
            return records

    def _write_records(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temp_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _sort_desc(records, field):
        return sorted(records, key=lambda item: item.get(field, ""), reverse=True)

    @staticmethod
    def _now():
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def list_documents(self):
        return self._sort_desc(self._read_records(self.document_path), "uploaded_at")

    def get_document_by_id(self, document_id):
        return next(
            (item for item in self._read_records(self.document_path) if item.get("id") == document_id), None
        )

    def get_document_by_hash(self, file_hash):
        return next(
            (item for item in self._read_records(self.document_path) if item.get("file_hash") == file_hash),
            None,
        )

    def add_document(self, record):
        with self._lock:
            records = self._read_records(self.document_path)
            new_record = deepcopy(record)
            new_record.setdefault("id", uuid4().hex)
            new_record.setdefault("uploaded_at", self._now())
            records.append(new_record)
            self._write_records(self.document_path, records)
            return new_record

    def update_document(self, document_id, patch):
        with self._lock:
            records = self._read_records(self.document_path)
            updated = None
            for record in records:
                if record.get("id") == document_id:
                    record.update(deepcopy(patch))
                    updated = record
                    break
            if updated is not None:
                self._write_records(self.document_path, records)
            return updated

    def delete_document(self, document_id):
        with self._lock:
            records = self._read_records(self.document_path)
            remaining = [record for record in records if record.get("id") != document_id]
            if len(remaining) == len(records):
                return False
            self._write_records(self.document_path, remaining)
            return True

    def list_qa_logs(self, limit=None):
        records = self._sort_desc(self._read_records(self.qa_path), "created_at")
        return records[:limit] if limit else records

    def list_session_logs(self, session_id, limit=None):
        records = [item for item in self._read_records(self.qa_path) if item.get("session_id") == session_id]
        records.sort(key=lambda item: item.get("created_at", ""))
        return records[-limit:] if limit else records

    def add_qa_log(self, record):
        with self._lock:
            records = self._read_records(self.qa_path)
            new_record = deepcopy(record)
            new_record.setdefault("id", uuid4().hex)
            new_record.setdefault("created_at", self._now())
            records.append(new_record)
            self._write_records(self.qa_path, records)
            return new_record

    def list_feedback(self):
        return self._sort_desc(self._read_records(self.feedback_path), "created_at")

    def get_feedback_by_qa_log_id(self, qa_log_id):
        return next(
            (item for item in self._read_records(self.feedback_path) if item.get("qa_log_id") == qa_log_id),
            None,
        )

    def upsert_feedback(self, qa_log_id, rating, comment, session_id):
        with self._lock:
            records = self._read_records(self.feedback_path)
            now = self._now()
            for record in records:
                if record.get("qa_log_id") == qa_log_id:
                    record.update({"rating": rating, "comment": comment, "updated_at": now})
                    self._write_records(self.feedback_path, records)
                    return record
            new_record = {
                "id": uuid4().hex,
                "qa_log_id": qa_log_id,
                "rating": rating,
                "comment": comment,
                "session_id": session_id,
                "created_at": now,
                "updated_at": now,
            }
            records.append(new_record)
            self._write_records(self.feedback_path, records)
            return new_record

    def get_stats(self):
        documents = self._read_records(self.document_path)
        qa_logs = self._read_records(self.qa_path)
        feedback = self._read_records(self.feedback_path)
        categories = {record.get("category") for record in documents if record.get("category")}
        rated = [record for record in feedback if record.get("rating") in {"helpful", "needs_improvement"}]
        helpful = [record for record in rated if record.get("rating") == "helpful"]
        return {
            "document_count": len(documents),
            "category_count": len(categories),
            "qa_count": len(qa_logs),
            "feedback_count": len(feedback),
            "helpful_rate": len(helpful) / len(rated) if rated else 0.0,
        }

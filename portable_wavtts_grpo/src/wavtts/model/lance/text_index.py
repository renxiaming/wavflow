import json
import os
import sqlite3
from pathlib import Path


class EmiliaTextIndex:
    """Random-access utt -> text lookup for Emilia Lance training."""

    def __init__(self, index_path: str | None = None, jsonl_path: str | None = None):
        self._cache: dict[str, str] = {}
        self._conn: sqlite3.Connection | None = None
        self._index_path: str | None = None
        self._jsonl_path: str | None = None

        if index_path:
            if not os.path.isfile(index_path):
                raise FileNotFoundError(
                    f"Emilia text sqlite index not found: {index_path}\n"
                    "Build it once with:\n"
                    "  python src/wavtts/train/datasets/build_emilia_text_index.py "
                    f"--jsonl_path {jsonl_path or '/home/work_nfs23/hkxie/data/Emilia/emilia_merged.jsonl'} "
                    f"--db_path {index_path}"
                )
            self._index_path = index_path
            return

        if jsonl_path and os.path.isfile(jsonl_path):
            self._jsonl_path = jsonl_path
            return

        raise FileNotFoundError(
            "Emilia text lookup requires datasets.text_index_path (recommended) "
            "or datasets.text_jsonl_path for small debug runs."
        )

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self._index_path:
                raise RuntimeError("sqlite index path is not configured")
            self._conn = sqlite3.connect(
                f"file:{self._index_path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
        return self._conn

    def __getstate__(self):
        return {
            "_cache": self._cache,
            "_index_path": self._index_path,
            "_jsonl_path": self._jsonl_path,
        }

    def __setstate__(self, state):
        self._cache = state["_cache"]
        self._index_path = state["_index_path"]
        self._jsonl_path = state["_jsonl_path"]
        self._conn = None

    def get_text(self, utt: str) -> str | None:
        if self._index_path is not None:
            row = self._get_conn().execute(
                "SELECT text FROM utt_text WHERE utt = ?", (utt,)
            ).fetchone()
            return row[0] if row else None

        if utt in self._cache:
            return self._cache[utt]

        with open(self._jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                sample_id = obj.get("id") or obj.get("utt")
                if sample_id == utt:
                    text = obj["text"]
                    self._cache[utt] = text
                    return text
        return None

    @staticmethod
    def build_sqlite(jsonl_path: str, db_path: str, batch_size: int = 10_000):
        db_path = str(Path(db_path))
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE utt_text (utt TEXT PRIMARY KEY, text TEXT NOT NULL, language TEXT)"
        )

        batch: list[tuple[str, str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                utt = obj.get("id") or obj.get("utt")
                if not utt:
                    continue
                batch.append((utt, obj["text"], (obj.get("language") or "").lower()))
                if len(batch) >= batch_size:
                    conn.executemany(
                        "INSERT OR REPLACE INTO utt_text (utt, text, language) VALUES (?, ?, ?)",
                        batch,
                    )
                    conn.commit()
                    batch.clear()

        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO utt_text (utt, text, language) VALUES (?, ?, ?)",
                batch,
            )
            conn.commit()

        # PRIMARY KEY on utt already provides the lookup index.
        conn.close()

    @staticmethod
    def build_sqlite_from_li(
        li_path: str,
        db_path: str,
        source_db_path: str,
        batch_size: int = 5_000,
    ):
        """Build a subset sqlite index from a Lance .li file and an existing full index."""
        db_path = str(Path(db_path))
        source_db_path = str(Path(source_db_path))
        if not os.path.isfile(source_db_path):
            raise FileNotFoundError(f"Source sqlite index not found: {source_db_path}")
        if not os.path.isfile(li_path):
            raise FileNotFoundError(f"Lance index file not found: {li_path}")

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)

        utts = []
        with open(li_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utts.append(line.split("|", 1)[0])

        src = sqlite3.connect(f"file:{source_db_path}?mode=ro", uri=True)
        dst = sqlite3.connect(db_path)
        dst.execute("PRAGMA journal_mode=WAL")
        has_language = bool(
            src.execute(
                "SELECT 1 FROM pragma_table_info('utt_text') WHERE name='language'"
            ).fetchone()
        )
        if has_language:
            dst.execute(
                "CREATE TABLE utt_text (utt TEXT PRIMARY KEY, text TEXT NOT NULL, language TEXT)"
            )
        else:
            dst.execute("CREATE TABLE utt_text (utt TEXT PRIMARY KEY, text TEXT NOT NULL)")

        missing = 0
        batch: list[tuple] = []
        for i in range(0, len(utts), batch_size):
            chunk = utts[i : i + batch_size]
            placeholders = ",".join("?" for _ in chunk)
            if has_language:
                rows = src.execute(
                    f"SELECT utt, text, language FROM utt_text WHERE utt IN ({placeholders})",
                    chunk,
                ).fetchall()
                found = {utt: (text, language) for utt, text, language in rows}
                for utt in chunk:
                    item = found.get(utt)
                    if item is None:
                        missing += 1
                        continue
                    batch.append((utt, item[0], item[1]))
                if batch:
                    dst.executemany(
                        "INSERT OR REPLACE INTO utt_text (utt, text, language) VALUES (?, ?, ?)",
                        batch,
                    )
            else:
                rows = src.execute(
                    f"SELECT utt, text FROM utt_text WHERE utt IN ({placeholders})",
                    chunk,
                ).fetchall()
                found = {utt: text for utt, text in rows}
                for utt in chunk:
                    text = found.get(utt)
                    if text is None:
                        missing += 1
                        continue
                    batch.append((utt, text))
                if batch:
                    dst.executemany(
                        "INSERT OR REPLACE INTO utt_text (utt, text) VALUES (?, ?)",
                        batch,
                    )
            dst.commit()
            batch.clear()

        src.close()
        dst.close()

        if missing:
            print(f"Warning: {missing} utts from {li_path} were missing in {source_db_path}")

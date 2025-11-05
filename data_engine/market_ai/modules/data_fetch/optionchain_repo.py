# market_ai/modules/data_fetch/optionchain_repo.py
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional
import logging, sqlite3, json
from datetime import date

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

SCHEMA = """
CREATE TABLE IF NOT EXISTS option_chain (
    underlying_id INTEGER NOT NULL,
    underlying_seg TEXT NOT NULL,
    as_of_date    TEXT NOT NULL,      -- 'YYYY-MM-DD' (entry day)
    expiry        TEXT NOT NULL,      -- 'YYYY-MM-DD' (contract expiry)
    strike        REAL NOT NULL,
    ce_json       TEXT,               -- JSON blob for CE quote+greeks at that time
    pe_json       TEXT,               -- JSON blob for PE quote+greeks at that time
    PRIMARY KEY (underlying_id, underlying_seg, as_of_date, expiry, strike)
);
CREATE INDEX IF NOT EXISTS idx_oc_lookup
    ON option_chain(underlying_id, underlying_seg, as_of_date, expiry);
"""

class OptionChainRepo:
    """Historical OC store backed by SQLite (or compatible)."""
    def __init__(self, db_path: str = "oc_history.sqlite"):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as cx:
            cx.executescript(SCHEMA)

    def upsert_many(self, rows: Iterable[dict]):
        with sqlite3.connect(self.db_path) as cx:
            cx.execute("PRAGMA journal_mode=WAL;")
            cur = cx.cursor()
            for r in rows:
                cur.execute(
                    """INSERT OR REPLACE INTO option_chain
                       (underlying_id, underlying_seg, as_of_date, expiry, strike, ce_json, pe_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        int(r["underlying_id"]), str(r["underlying_seg"]),
                        str(r["as_of_date"]), str(r["expiry"]),
                        float(r["strike"]),
                        json.dumps(r.get("ce") or {}), json.dumps(r.get("pe") or {}),
                    )
                )
            cx.commit()

    def get_chain(self, underlying_id: int, underlying_seg: str, as_of_date: str, expiry: str) -> Dict[str, Any]:
        """Return normalized dict: { '<strike>': {'ce': {...}, 'pe': {...}}, ... }"""
        out: Dict[str, Any] = {}
        with sqlite3.connect(self.db_path) as cx:
            cur = cx.cursor()
            cur.execute(
                """SELECT strike, ce_json, pe_json
                   FROM option_chain
                   WHERE underlying_id=? AND underlying_seg=? AND as_of_date=? AND expiry=?
                   ORDER BY strike ASC""",
                (int(underlying_id), str(underlying_seg), str(as_of_date), str(expiry))
            )
            for strike, ce_j, pe_j in cur.fetchall():
                try:
                    out[str(float(strike))] = {"ce": json.loads(ce_j or "{}"), "pe": json.loads(pe_j or "{}")}
                except Exception:
                    continue
        return out

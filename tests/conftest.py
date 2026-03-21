import sqlite3
import pytest


@pytest.fixture
def db():
    """In-memory SQLite database with BESPOKE schema for testing."""
    from bespoke.db.init import _ensure_schema
    import sqlite_vec

    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema(conn)
    conn.commit()
    yield conn
    conn.close()

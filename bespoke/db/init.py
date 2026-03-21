"""Database initialization for BESPOKE."""

import sqlite3
import sqlite_vec
from pathlib import Path


DB_PATH = Path.home() / ".bespoke" / "bespoke.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"
EMBEDDING_DIM = 768  # EmbeddingGemma 300M output dimension


_initialized = False


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Get a database connection with sqlite-vec loaded.

    Auto-initializes the schema on first use.
    """
    global _initialized
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Enable WAL mode for concurrent access
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")

    conn.row_factory = sqlite3.Row

    if not _initialized:
        _ensure_schema(conn)
        _initialized = True

    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Apply schema and migrations if tables are missing."""
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {row['name'] for row in tables}

    required = {'interactions', 'training_pairs', 'meta_patterns',
                'experiments', 'pipeline_state'}

    if not required.issubset(table_names):
        schema_sql = SCHEMA_PATH.read_text()
        conn.executescript(schema_sql)

    # Vector table
    try:
        conn.execute("SELECT interaction_embedding FROM vec_interactions LIMIT 0")
    except Exception:
        conn.execute("DROP TABLE IF EXISTS vec_interactions")
        conn.execute(f"""
            CREATE VIRTUAL TABLE vec_interactions USING vec0(
                interaction_embedding float[{EMBEDDING_DIM}]
            )
        """)

    # Column migrations
    for col in ["content_hash TEXT", "user_followup TEXT",
                "stage2a_fail_count INTEGER DEFAULT 0"]:
        try:
            conn.execute(f"ALTER TABLE interactions ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_interactions_content_hash ON interactions(content_hash)")

    conn.commit()


def init_database(db_path: Path = DB_PATH) -> None:
    """Initialize the database with schema and vector table."""
    conn = get_connection(db_path)  # auto-applies schema
    conn.close()
    print(f"Database initialized at {db_path}")


def verify_database(db_path: Path = DB_PATH) -> bool:
    """Verify the database is properly initialized."""
    conn = get_connection(db_path)

    # Check tables exist
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {row['name'] for row in tables}

    required = {'interactions', 'training_pairs', 'meta_patterns',
                'experiments', 'pipeline_state'}

    missing = required - table_names
    if missing:
        print(f"Missing tables: {missing}")
        return False

    # Check vec table
    vec_version = conn.execute("SELECT vec_version()").fetchone()[0]
    print(f"sqlite-vec version: {vec_version}")

    # Check vec_interactions exists
    try:
        conn.execute("SELECT count(*) FROM vec_interactions")
        print("Vector table: OK")
    except Exception as e:
        print(f"Vector table error: {e}")
        return False

    conn.close()
    print("Database verification: PASSED")
    return True


if __name__ == "__main__":
    init_database()
    verify_database()

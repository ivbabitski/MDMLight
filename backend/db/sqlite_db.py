import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

DB_PATH = os.environ.get("DB_PATH", "/data/app.db")

def _ensure_db_dir() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    def _norm_sql(s: object) -> str:
        if s is None:
            return ""
        v = str(s).strip().lower()

        # Strip punctuation by converting non-alnum chars to spaces,
        # then collapse whitespace.
        cleaned = []
        for ch in v:
            if ch.isalnum() or ch.isspace():
                cleaned.append(ch)
            else:
                cleaned.append(" ")
        v = "".join(cleaned)
        v = " ".join(v.split())
        return v

    conn.create_function("norm", 1, _norm_sql)

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_app_user_id(conn: sqlite3.Connection, table_name: str) -> None:
    """
    Ensure app_user_id exists for multi-user ownership.

    NOTE: This does NOT touch source audit fields like created_by/updated_by.
    """
    info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    cols = {r["name"] for r in info}
    if "app_user_id" not in cols:
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN app_user_id INTEGER REFERENCES users(id)"
        )

def _ensure_column(conn: sqlite3.Connection, table_name: str, col_name: str, col_sql: str) -> None:
    info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    cols = {r["name"] for r in info}
    if col_name not in cols:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_sql}")


def init_users() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

def create_user(username: str, email: str, password_hash: str) -> None:
    init_users()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )

def get_user_by_username(username: str):
    init_users()
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ?",
            (username,),
        )
        return cur.fetchone()

def update_user_password(username: str, password_hash: str) -> bool:
    init_users()
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (password_hash, username),
        )
        return cur.rowcount > 0


def list_mdm_models(include_deleted: bool = False):
    init_mdm_models()
    where = "" if include_deleted else "WHERE deleted_at IS NULL"
    with get_conn() as conn:
        cur = conn.execute(
            f"""
            SELECT
                id, model_key, model_name, config_json,
                owner_user_id, owner_username,
                app_user_id,
                created_at, created_by,
                updated_at, updated_by,
                deleted_at, deleted_by
            FROM mdm_models
            {where}
            ORDER BY created_at DESC
            """
        )
        return cur.fetchall()


def list_mdm_models(include_deleted: bool = False):
    init_mdm_models()
    where = "" if include_deleted else "WHERE deleted_at IS NULL"
    with get_conn() as conn:
        cur = conn.execute(
            f"""
            SELECT
                id, model_key, model_name, config_json,
                owner_user_id, owner_username,
                app_user_id,
                created_at, created_by,
                updated_at, updated_by,
                deleted_at, deleted_by
            FROM mdm_models
            {where}
            ORDER BY created_at DESC
            """
        )
        return cur.fetchall()


def get_mdm_model(model_id_or_key: str, include_deleted: bool = False):
    init_mdm_models()
    filt = "" if include_deleted else "AND deleted_at IS NULL"
    with get_conn() as conn:
        cur = conn.execute(
            f"""
            SELECT
                id, model_key, model_name, config_json,
                owner_user_id, owner_username,
                app_user_id,
                created_at, created_by,
                updated_at, updated_by,
                deleted_at, deleted_by
            FROM mdm_models
            WHERE (id = ? OR model_key = ?)
            {filt}
            LIMIT 1
            """,
            (model_id_or_key, model_id_or_key),
        )
        return cur.fetchone()


def update_mdm_model(model_id: str, model_name, config_json, actor: str, now: str) -> bool:
    init_mdm_models()

    sets = []
    params = []

    if model_name is not None:
        sets.append("model_name = ?")
        params.append(model_name)

    if config_json is not None:
        sets.append("config_json = ?")
        params.append(config_json)

    sets.append("updated_at = ?")
    sets.append("updated_by = ?")
    params.append(now)
    params.append(actor)

    if not sets:
        return False

    params.append(model_id)

    with get_conn() as conn:
        cur = conn.execute(
            f"""
            UPDATE mdm_models
            SET {", ".join(sets)}
            WHERE id = ?
              AND deleted_at IS NULL
            """,
            tuple(params),
        )
        return cur.rowcount > 0


def init_mdm_models() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mdm_models (
                id TEXT PRIMARY KEY,
                model_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                config_json TEXT NOT NULL,

                owner_user_id INTEGER NOT NULL,
                owner_username TEXT NOT NULL,

                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                updated_at TEXT,
                updated_by TEXT,

                deleted_at TEXT,
                deleted_by TEXT,
                app_user_id INTEGER REFERENCES users(id)
            )
            """
        )
        _ensure_app_user_id(conn, "mdm_models")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_model_name ON mdm_models(model_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_model_key ON mdm_models(model_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_deleted_at ON mdm_models(deleted_at)"
        )

        try:
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_mdm_models_active_model_key
                ON mdm_models(model_key)
                WHERE deleted_at IS NULL
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_mdm_models_active_model_name
                ON mdm_models(model_name)
                WHERE deleted_at IS NULL
                """
            )
        except sqlite3.OperationalError:
            pass


def create_mdm_model(
    model_key: str,
    model_name: str,
    config_json: str,
    owner_user_id: int,
    actor: str,
    now: str,
) -> str:
    init_mdm_models()
    import uuid

    model_id = str(uuid.uuid4())

    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT 1
            FROM mdm_models
            WHERE deleted_at IS NULL
              AND (model_key = ? OR model_name = ?)
            LIMIT 1
            """,
            (model_key, model_name),
        )
        if cur.fetchone():
            raise sqlite3.IntegrityError("active model_key or model_name already exists")

        conn.execute(
            """
            INSERT INTO mdm_models (
                id, model_key, model_name, config_json,
                owner_user_id, owner_username,
                created_at, created_by,
                updated_at, updated_by,
                deleted_at, deleted_by,
                app_user_id
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                NULL, NULL,
                NULL, NULL,
                ?
            )
            """,
            (model_id, model_key, model_name, config_json, owner_user_id, actor, now, actor, owner_user_id),
        )

    return model_id


def get_mdm_model_by_id(model_id: str, include_deleted: bool = False):
    init_mdm_models()
    filt = "" if include_deleted else "AND deleted_at IS NULL"

    with get_conn() as conn:
        cur = conn.execute(
            f"""
            SELECT
                id, model_key, model_name, config_json,
                owner_user_id, owner_username,
                app_user_id,
                created_at, created_by,
                updated_at, updated_by,
                deleted_at, deleted_by
            FROM mdm_models
            WHERE id = ?
            {filt}
            LIMIT 1
            """,
            (model_id,),
        )
        return cur.fetchone()


def get_mdm_model_by_name(model_name: str, include_deleted: bool = False):
    init_mdm_models()
    filt = "" if include_deleted else "AND deleted_at IS NULL"

    with get_conn() as conn:
        cur = conn.execute(
            f"""
            SELECT
                id, model_key, model_name, config_json,
                owner_user_id, owner_username,
                app_user_id,
                created_at, created_by,
                updated_at, updated_by,
                deleted_at, deleted_by
            FROM mdm_models
            WHERE model_name = ?
            {filt}
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (model_name,),
        )
        return cur.fetchone()


def soft_delete_mdm_model(model_id: str, actor: str, now: str) -> bool:
    init_mdm_models()

    with get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE mdm_models
            SET
                deleted_at = ?,
                deleted_by = ?,
                updated_at = ?,
                updated_by = ?
            WHERE id = ?
              AND deleted_at IS NULL
            """,
            (now, actor, now, actor, model_id),
        )
        return cur.rowcount > 0

def init_mdm_models() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mdm_models (
                id TEXT PRIMARY KEY,
                model_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                config_json TEXT NOT NULL,

                owner_user_id INTEGER NOT NULL,
                owner_username TEXT NOT NULL,

                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                updated_at TEXT,
                updated_by TEXT,

                deleted_at TEXT,
                deleted_by TEXT,
                app_user_id INTEGER REFERENCES users(id)
            )
            """
        )
        _ensure_app_user_id(conn, "mdm_models")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_model_name ON mdm_models(model_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_model_key ON mdm_models(model_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_deleted_at ON mdm_models(deleted_at)"
        )

        try:
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_mdm_models_active_model_key
                ON mdm_models(model_key)
                WHERE deleted_at IS NULL
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS ux_mdm_models_active_model_name
                ON mdm_models(model_name)
                WHERE deleted_at IS NULL
                """
            )
        except sqlite3.OperationalError:
            pass


def create_mdm_model(
    model_key: str,
    model_name: str,
    config_json: str,
    owner_user_id: int,
    actor: str,
    now: str,
) -> str:
    init_mdm_models()
    import uuid

    model_id = str(uuid.uuid4())

    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT 1
            FROM mdm_models
            WHERE deleted_at IS NULL
              AND (model_key = ? OR model_name = ?)
            LIMIT 1
            """,
            (model_key, model_name),
        )
        if cur.fetchone():
            raise sqlite3.IntegrityError("active model_key or model_name already exists")

        conn.execute(
            """
            INSERT INTO mdm_models (
                id, model_key, model_name, config_json,
                owner_user_id, owner_username,
                created_at, created_by,
                updated_at, updated_by,
                deleted_at, deleted_by,
                app_user_id
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                NULL, NULL,
                NULL, NULL,
                ?
            )
            """,
            (model_id, model_key, model_name, config_json, owner_user_id, actor, now, actor, owner_user_id),
        )

    return model_id


def get_mdm_model_by_id(model_id: str, include_deleted: bool = False):
    init_mdm_models()
    filt = "" if include_deleted else "AND deleted_at IS NULL"

    with get_conn() as conn:
        cur = conn.execute(
            f"""
            SELECT
                id, model_key, model_name, config_json,
                owner_user_id, owner_username,
                app_user_id,
                created_at, created_by,
                updated_at, updated_by,
                deleted_at, deleted_by
            FROM mdm_models
            WHERE id = ?
            {filt}
            LIMIT 1
            """,
            (model_id,),
        )
        return cur.fetchone()


def get_mdm_model_by_name(model_name: str, include_deleted: bool = False):
    init_mdm_models()
    filt = "" if include_deleted else "AND deleted_at IS NULL"

    with get_conn() as conn:
        cur = conn.execute(
            f"""
            SELECT
                id, model_key, model_name, config_json,
                owner_user_id, owner_username,
                app_user_id,
                created_at, created_by,
                updated_at, updated_by,
                deleted_at, deleted_by
            FROM mdm_models
            WHERE model_name = ?
            {filt}
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (model_name,),
        )
        return cur.fetchone()


def soft_delete_mdm_model(model_id: str, actor: str, now: str) -> bool:
    init_mdm_models()

    with get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE mdm_models
            SET
                deleted_at = ?,
                deleted_by = ?,
                updated_at = ?,
                updated_by = ?
            WHERE id = ?
              AND deleted_at IS NULL
            """,
            (now, actor, now, actor, model_id),
        )
        return cur.rowcount > 0



def init_source_input() -> None:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='source_input'"
        )
        exists = cur.fetchone() is not None

        needs_rebuild = False
        cols = []
        pk_cols = []

        if exists:
            info = conn.execute("PRAGMA table_info(source_input)").fetchall()
            cols = [r["name"] for r in info]
            pk_cols = [r["name"] for r in sorted(info, key=lambda x: x["pk"]) if r["pk"]]

            if "id" in cols:
                needs_rebuild = True
            elif pk_cols != ["app_user_id", "source_name", "source_id"]:
                needs_rebuild = True
            elif cols and cols[0] != "app_user_id":
                needs_rebuild = True

        if exists and needs_rebuild:
            conn.execute("ALTER TABLE source_input RENAME TO source_input_old")

            conn.execute(
                """
                CREATE TABLE source_input (
                    app_user_id INTEGER NOT NULL REFERENCES users(id),
                    source_name TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    f01 TEXT, f02 TEXT, f03 TEXT, f04 TEXT, f05 TEXT,
                    f06 TEXT, f07 TEXT, f08 TEXT, f09 TEXT, f10 TEXT,
                    f11 TEXT, f12 TEXT, f13 TEXT, f14 TEXT, f15 TEXT,
                    f16 TEXT, f17 TEXT, f18 TEXT, f19 TEXT, f20 TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    updated_at TEXT,
                    updated_by TEXT,
                    PRIMARY KEY (app_user_id, source_name, source_id)
                )
                """
            )

            wanted = [
                "app_user_id",
                "source_name",
                "source_id",
                "f01","f02","f03","f04","f05","f06","f07","f08","f09","f10",
                "f11","f12","f13","f14","f15","f16","f17","f18","f19","f20",
                "created_at",
                "created_by",
                "updated_at",
                "updated_by",
            ]

            have = set(cols)
            copy_cols = [c for c in wanted if c in have]
            if copy_cols:
                cols_sql = ",".join(copy_cols)
                conn.execute(
                    f"INSERT OR REPLACE INTO source_input ({cols_sql}) SELECT {cols_sql} FROM source_input_old"
                )

            conn.execute("DROP TABLE source_input_old")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_input (
                app_user_id INTEGER NOT NULL REFERENCES users(id),
                source_name TEXT NOT NULL,
                source_id TEXT NOT NULL,
                f01 TEXT, f02 TEXT, f03 TEXT, f04 TEXT, f05 TEXT,
                f06 TEXT, f07 TEXT, f08 TEXT, f09 TEXT, f10 TEXT,
                f11 TEXT, f12 TEXT, f13 TEXT, f14 TEXT, f15 TEXT,
                f16 TEXT, f17 TEXT, f18 TEXT, f19 TEXT, f20 TEXT,
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                updated_at TEXT,
                updated_by TEXT,
                PRIMARY KEY (app_user_id, source_name, source_id)
            )
            """
        )

        # old uniqueness blocks multi-user; drop it
        conn.execute("DROP INDEX IF EXISTS ux_source_input_source_name_id")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_input_source_id ON source_input(source_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_input_user_source ON source_input(app_user_id, source_name, source_id)"
        )


def init_match_job() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS match_job (
              job_id TEXT PRIMARY KEY,

              status TEXT NOT NULL,          -- queued | running | completed | failed
              model_id TEXT,
              model_json TEXT NOT NULL,

              message TEXT,
              exceptions_json TEXT,

              total_records INTEGER,
              total_buckets INTEGER,
              total_pairs_scored INTEGER,
              total_matches INTEGER,
              total_clusters INTEGER,

              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              started_at TEXT,
              finished_at TEXT,

              app_user_id INTEGER REFERENCES users(id)
            )
            """
        )
        _ensure_app_user_id(conn, "match_job")
        _ensure_column(conn, "match_job", "model_id", "model_id TEXT")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_match_job_status ON match_job(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_match_job_user_model_status ON match_job(app_user_id, model_id, status)"
        )


def init_recon_cluster():
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recon_cluster'"
        )
        exists = cur.fetchone() is not None

        needs_rebuild = False
        if exists:
            schema_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='recon_cluster'"
            ).fetchone()
            schema_sql = str(schema_row["sql"] or "") if schema_row else ""
            if "REFERENCES app_user" in schema_sql:
                needs_rebuild = True
            else:
                fks = conn.execute("PRAGMA foreign_key_list(recon_cluster)").fetchall()
                for fk in fks:
                    if fk["from"] == "app_user_id" and fk["table"] != "users":
                        needs_rebuild = True
                        break

        if exists and needs_rebuild:
            conn.execute("ALTER TABLE recon_cluster RENAME TO recon_cluster_old")
            conn.execute("DROP INDEX IF EXISTS ix_recon_cluster_cluster_id")
            conn.execute("DROP INDEX IF EXISTS ix_recon_cluster_user_model")
            conn.execute("DROP INDEX IF EXISTS ix_recon_cluster_user_model_cluster")
            conn.execute("DROP INDEX IF EXISTS ix_recon_cluster_user_model_status")
            conn.execute("DROP INDEX IF EXISTS ux_recon_cluster_identity")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recon_cluster(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id TEXT NOT NULL,
                model_id TEXT,
                model_name TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_id TEXT NOT NULL,

                f01 TEXT, f02 TEXT, f03 TEXT, f04 TEXT, f05 TEXT,
                f06 TEXT, f07 TEXT, f08 TEXT, f09 TEXT, f10 TEXT,
                f11 TEXT, f12 TEXT, f13 TEXT, f14 TEXT, f15 TEXT,
                f16 TEXT, f17 TEXT, f18 TEXT, f19 TEXT, f20 TEXT,

                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                updated_at TEXT,
                updated_by TEXT,

                match_status TEXT DEFAULT 'match',
                match_score REAL,
                app_user_id INTEGER REFERENCES users(id)
            )
            """
        )

        if exists and needs_rebuild:
            conn.execute(
                """
                INSERT INTO recon_cluster (
                    id,
                    cluster_id,
                    model_id,
                    model_name,
                    source_name,
                    source_id,

                    f01, f02, f03, f04, f05,
                    f06, f07, f08, f09, f10,
                    f11, f12, f13, f14, f15,
                    f16, f17, f18, f19, f20,

                    created_at,
                    created_by,
                    updated_at,
                    updated_by,

                    match_status,
                    match_score,
                    app_user_id
                )
                SELECT
                    id,
                    cluster_id,
                    model_id,
                    model_name,
                    source_name,
                    source_id,

                    f01, f02, f03, f04, f05,
                    f06, f07, f08, f09, f10,
                    f11, f12, f13, f14, f15,
                    f16, f17, f18, f19, f20,

                    created_at,
                    created_by,
                    updated_at,
                    updated_by,

                    match_status,
                    match_score,
                    app_user_id
                FROM recon_cluster_old
                """
            )
            conn.execute("DROP TABLE recon_cluster_old")
        _ensure_app_user_id(conn, "recon_cluster")
        _ensure_column(conn, "recon_cluster", "model_id", "model_id TEXT")
        _ensure_column(conn, "recon_cluster", "model_name", "model_name TEXT")
        _ensure_column(conn, "recon_cluster", "cluster_id", "cluster_id TEXT")
        _ensure_column(conn, "recon_cluster", "source_name", "source_name TEXT")
        _ensure_column(conn, "recon_cluster", "source_id", "source_id TEXT")
        _ensure_column(conn, "recon_cluster", "created_at", "created_at TEXT")
        _ensure_column(conn, "recon_cluster", "created_by", "created_by TEXT")
        _ensure_column(conn, "recon_cluster", "updated_at", "updated_at TEXT")
        _ensure_column(conn, "recon_cluster", "updated_by", "updated_by TEXT")
        _ensure_column(conn, "recon_cluster", "match_status", "match_status TEXT DEFAULT 'match'")
        _ensure_column(conn, "recon_cluster", "match_score", "match_score REAL")
        for i in range(1, 21):
            c = f"f{str(i).zfill(2)}"
            _ensure_column(conn, "recon_cluster", c, f"{c} TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS ix_recon_cluster_cluster_id ON recon_cluster(cluster_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_recon_cluster_user_model ON recon_cluster(app_user_id, model_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_recon_cluster_user_model_cluster ON recon_cluster(app_user_id, model_id, cluster_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_recon_cluster_user_model_status ON recon_cluster(app_user_id, model_id, match_status)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_recon_cluster_identity ON recon_cluster(app_user_id, model_id, source_name, source_id)"
        )



def init_cluster_map() -> None:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='cluster_map'"
        )
        exists = cur.fetchone() is not None

        needs_rebuild = False
        cols = []
        pk_cols = []

        desired_pk = ["app_user_id", "model_id", "source_name", "source_id"]

        if exists:
            info = conn.execute("PRAGMA table_info(cluster_map)").fetchall()
            cols = [r["name"] for r in info]
            pk_cols = [r["name"] for r in sorted(info, key=lambda x: x["pk"]) if r["pk"]]

            if "model_id" not in cols:
                needs_rebuild = True
            elif pk_cols != desired_pk:
                needs_rebuild = True

        if exists and needs_rebuild:
            conn.execute("ALTER TABLE cluster_map RENAME TO cluster_map_old")

            conn.execute(
                """
                CREATE TABLE cluster_map (
                  app_user_id INTEGER NOT NULL REFERENCES users(id),
                  model_id TEXT NOT NULL DEFAULT 'legacy',

                  source_name TEXT NOT NULL,
                  source_id   TEXT NOT NULL,
                  cluster_id  TEXT NOT NULL,

                  first_seen_at TEXT NOT NULL,
                  last_seen_at  TEXT NOT NULL,

                  PRIMARY KEY (app_user_id, model_id, source_name, source_id)
                )
                """
            )

            # Legacy rows have no model_id; store them under model_id='legacy'.
            # Rows without app_user_id are invalid and are skipped.
            conn.execute(
                """
                INSERT OR REPLACE INTO cluster_map (
                  app_user_id, model_id,
                  source_name, source_id,
                  cluster_id,
                  first_seen_at, last_seen_at
                )
                SELECT
                  app_user_id, 'legacy',
                  source_name, source_id,
                  cluster_id,
                  first_seen_at, last_seen_at
                FROM cluster_map_old
                WHERE app_user_id IS NOT NULL
                """
            )

            conn.execute("DROP TABLE cluster_map_old")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cluster_map (
              app_user_id INTEGER NOT NULL REFERENCES users(id),
              model_id TEXT NOT NULL DEFAULT 'legacy',

              source_name TEXT NOT NULL,
              source_id   TEXT NOT NULL,
              cluster_id  TEXT NOT NULL,

              first_seen_at TEXT NOT NULL,
              last_seen_at  TEXT NOT NULL,

              PRIMARY KEY (app_user_id, model_id, source_name, source_id)
            )
            """
        )

        _ensure_column(conn, "cluster_map", "model_id", "model_id TEXT")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_cluster_map_cluster_id ON cluster_map(cluster_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_cluster_map_user_model ON cluster_map(app_user_id, model_id)"
        )


def init_golden_record() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS golden_record (
              master_id TEXT PRIMARY KEY,
              job_id TEXT NOT NULL,
              model_id TEXT,

              source_name TEXT NOT NULL DEFAULT 'MDM',

              match_threshold REAL NOT NULL,
              survivorship_json TEXT NOT NULL,
              representative_record_id TEXT NOT NULL,
              representative_source_name TEXT,
              representative_source_id TEXT,
              lineage_json TEXT,

              f01 TEXT, f02 TEXT, f03 TEXT, f04 TEXT, f05 TEXT,
              f06 TEXT, f07 TEXT, f08 TEXT, f09 TEXT, f10 TEXT,
              f11 TEXT, f12 TEXT, f13 TEXT, f14 TEXT, f15 TEXT,
              f16 TEXT, f17 TEXT, f18 TEXT, f19 TEXT, f20 TEXT,

              created_at TEXT NOT NULL,
              created_by TEXT NOT NULL,
              updated_at TEXT,
              updated_by TEXT,

              app_user_id INTEGER REFERENCES users(id)
            )
            """
        )
        _ensure_app_user_id(conn, "golden_record")
        _ensure_column(conn, "golden_record", "model_id", "model_id TEXT")
        _ensure_column(conn, "golden_record", "job_id", "job_id TEXT")
        _ensure_column(conn, "golden_record", "source_name", "source_name TEXT")
        _ensure_column(conn, "golden_record", "match_threshold", "match_threshold REAL")
        _ensure_column(conn, "golden_record", "survivorship_json", "survivorship_json TEXT")
        _ensure_column(conn, "golden_record", "representative_record_id", "representative_record_id TEXT")
        _ensure_column(conn, "golden_record", "representative_source_name", "representative_source_name TEXT")
        _ensure_column(conn, "golden_record", "representative_source_id", "representative_source_id TEXT")
        _ensure_column(conn, "golden_record", "lineage_json", "lineage_json TEXT")
        _ensure_column(conn, "golden_record", "created_at", "created_at TEXT")
        _ensure_column(conn, "golden_record", "created_by", "created_by TEXT")
        _ensure_column(conn, "golden_record", "updated_at", "updated_at TEXT")
        _ensure_column(conn, "golden_record", "updated_by", "updated_by TEXT")
        for i in range(1, 21):
            c = f"f{str(i).zfill(2)}"
            _ensure_column(conn, "golden_record", c, f"{c} TEXT")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_golden_record_job_id ON golden_record(job_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_golden_record_user_model ON golden_record(app_user_id, model_id)"
        )


def init_match_exception() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS match_exception (
              job_id TEXT NOT NULL,
              model_id TEXT,

              record_id TEXT NOT NULL,
              source_name TEXT NOT NULL,
              source_id TEXT NOT NULL,

              candidate_cluster_id TEXT NOT NULL,
              candidate_record_id TEXT,
              candidate_source_name TEXT,
              candidate_source_id TEXT,
              score REAL,

              reason TEXT NOT NULL,
              details_json TEXT,

              created_at TEXT NOT NULL,
              resolved_at TEXT,
              resolved_by TEXT,

              app_user_id INTEGER REFERENCES users(id),
              PRIMARY KEY (job_id, record_id, candidate_cluster_id)
            )
            """
        )
        _ensure_app_user_id(conn, "match_exception")
        _ensure_column(conn, "match_exception", "model_id", "model_id TEXT")
        _ensure_column(conn, "match_exception", "candidate_source_name", "candidate_source_name TEXT")
        _ensure_column(conn, "match_exception", "candidate_source_id", "candidate_source_id TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS ix_match_exception_job ON match_exception(job_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_match_exception_candidate_cluster ON match_exception(candidate_cluster_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_match_exception_user_model ON match_exception(app_user_id, model_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_match_exception_user_model_record ON match_exception(app_user_id, model_id, source_name, source_id)")



def init_all_tables() -> None:
    # existing
    init_users()
    init_mdm_models()
    init_source_input()

    # new
    init_match_job()
    init_recon_cluster()
    init_cluster_map()
    init_golden_record()
    init_match_exception()





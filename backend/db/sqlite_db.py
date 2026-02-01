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


def init_mdm_models() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mdm_models (
                id TEXT PRIMARY KEY,
                model_key TEXT NOT NULL UNIQUE,
                model_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
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
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_deleted_at ON mdm_models(deleted_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_created_at ON mdm_models(created_at)"
        )


def create_mdm_model(model_key: str, model_name: str, config_json: str, actor: str, now: str) -> str:
    init_mdm_models()
    import uuid

    model_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO mdm_models (
                id, model_key, model_name, config_json,
                created_at, created_by,
                updated_at, updated_by,
                deleted_at, deleted_by,
                app_user_id
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?,
                NULL, NULL,
                NULL, NULL,
                ?
            )
            """,
            (model_id, model_key, model_name, config_json, now, actor),
        )
    return model_id


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
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_owner_user_id ON mdm_models(owner_user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_model_name ON mdm_models(model_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mdm_models_deleted_at ON mdm_models(deleted_at)"
        )

        # Uniqueness among ACTIVE models (soft delete still allows re-create).
        # If partial indexes aren't supported, app still works — you just won't get uniqueness enforcement.
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
            elif pk_cols != ["source_id"]:
                needs_rebuild = True
            elif cols and cols[0] != "source_id":
                needs_rebuild = True

        if exists and needs_rebuild:
            conn.execute("ALTER TABLE source_input RENAME TO source_input_old")

            conn.execute(
                """
                CREATE TABLE source_input (
                    source_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
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


            wanted = [
                "source_id",
                "source_name",
                "f01","f02","f03","f04","f05","f06","f07","f08","f09","f10",
                "f11","f12","f13","f14","f15","f16","f17","f18","f19","f20",
                "created_at",
                "created_by",
                "updated_at",
                "updated_by",
                "app_user_id",
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
                source_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
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
        _ensure_app_user_id(conn, "source_input")


        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_source_input_source_name_id ON source_input(source_name, source_id)"
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_input_source_id ON source_input(source_id)"
        )

def init_match_job() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS match_job (
              job_id TEXT PRIMARY KEY,

              status TEXT NOT NULL,          -- queued | running | completed | failed
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

        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_match_job_status ON match_job(status)"
        )


def init_recon_cluster() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recon_cluster (
              job_id TEXT NOT NULL,
              cluster_id TEXT NOT NULL,

              record_id TEXT NOT NULL,
              source_name TEXT NOT NULL,
              source_id TEXT NOT NULL,

              cluster_size INTEGER NOT NULL,
              is_representative INTEGER NOT NULL DEFAULT 0,

              created_at TEXT NOT NULL,

              app_user_id INTEGER REFERENCES users(id),
              PRIMARY KEY (job_id, record_id)
            )
            """
        )
        _ensure_app_user_id(conn, "recon_cluster")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_recon_cluster_job_cluster ON recon_cluster(job_id, cluster_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_recon_cluster_cluster_id ON recon_cluster(cluster_id)"
        )


def init_cluster_map() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cluster_map (
              source_name TEXT NOT NULL,
              source_id   TEXT NOT NULL,
              cluster_id  TEXT NOT NULL,

              first_seen_at TEXT NOT NULL,
              last_seen_at  TEXT NOT NULL,

              app_user_id INTEGER REFERENCES users(id),
              PRIMARY KEY (source_name, source_id)
            )
            """
        )
        _ensure_app_user_id(conn, "cluster_map")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_cluster_map_cluster_id ON cluster_map(cluster_id)"
        )


def init_golden_record() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS golden_record (
              master_id TEXT PRIMARY KEY,
              job_id TEXT NOT NULL,

              source_name TEXT NOT NULL DEFAULT 'MDM',

              match_threshold REAL NOT NULL,
              survivorship_json TEXT NOT NULL,
              representative_record_id TEXT NOT NULL,
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

        conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_golden_record_job_id ON golden_record(job_id)"
        )


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




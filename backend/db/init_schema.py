# backend/db/init_schema.py
from . import sqlite_db

def main() -> None:
    sqlite_db.init_all_tables()
    print("OK: sqlite schema is ready")

if __name__ == "__main__":
    main()

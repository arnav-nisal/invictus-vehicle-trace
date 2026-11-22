"""Database Access Layer for Vehicle Registry.

Provides thread-safe SQLite connection helpers, auto-initialization/seeding
checks for fresh git clones, and strongly typed RegistryVehicle queries.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

from backend.schemas import RegistryVehicle

logger = logging.getLogger("vehicle_trace.database")

# Project root directory and default database location
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "registry.db"


def ensure_db_exists(db_path: Optional[Path] = None) -> Path:
    """Ensure the SQLite registry database exists; auto-seed if missing.

    Because *.db is in .gitignore, fresh clones do not have registry.db.
    This guarantees zero crash on startup by running seed_registry.py.
    """
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH

    # Check if DB file exists and has the vehicles table with rows
    needs_seeding = False
    if not target_path.exists() or target_path.stat().st_size == 0:
        needs_seeding = True
    else:
        try:
            with sqlite3.connect(str(target_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='vehicles';")
                table_exists = cursor.fetchone()[0] > 0
                if not table_exists:
                    needs_seeding = True
                else:
                    cursor.execute("SELECT count(*) FROM vehicles;")
                    count = cursor.fetchone()[0]
                    if count == 0:
                        needs_seeding = True
        except Exception as e:
            logger.warning(f"Database check failed ({e}), triggering re-seed.")
            needs_seeding = True

    if needs_seeding:
        logger.info(f"Database missing or uninitialized at {target_path}. Running automated seeder...")
        # Ensure root directory is on sys.path for seed_registry import
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        from scripts.seed_registry import seed_database
        target_path.parent.mkdir(parents=True, exist_ok=True)
        seed_database(target_path, synthetic_target_count=1000)
        logger.info(f"Automated seeding complete: {target_path}")

    return target_path


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory configured."""
    target_path = ensure_db_exists(db_path)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_total_vehicles_count(db_path: Optional[Path] = None) -> int:
    """Return the total number of vehicle records registered in SQLite."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM vehicles;")
        count = cursor.fetchone()[0]
        return int(count)
    finally:
        conn.close()


def get_all_registry_vehicles(db_path: Optional[Path] = None) -> List[RegistryVehicle]:
    """Fetch all registered vehicles and map into Pydantic RegistryVehicle schemas."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT plate_number, body_type, colour, make, model, rto_city, synthetic_owner_id
            FROM vehicles;
        """)
        rows = cursor.fetchall()
        return [
            RegistryVehicle(
                plate_number=row["plate_number"],
                body_type=row["body_type"],
                colour=row["colour"],
                make=row["make"],
                model=row["model"],
                rto_city=row["rto_city"],
                synthetic_owner_id=row["synthetic_owner_id"],
            )
            for row in rows
        ]
    finally:
        conn.close()


def get_vehicle_by_plate(plate_number: str, db_path: Optional[Path] = None) -> Optional[RegistryVehicle]:
    """Retrieve a single vehicle record by plate number (case-insensitive)."""
    clean_plate = plate_number.strip().upper().replace(" ", "").replace("-", "")
    conn = get_db_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT plate_number, body_type, colour, make, model, rto_city, synthetic_owner_id
            FROM vehicles
            WHERE UPPER(REPLACE(REPLACE(plate_number, ' ', ''), '-', '')) = ?;
        """, (clean_plate,))
        row = cursor.fetchone()
        if not row:
            return None
        return RegistryVehicle(
            plate_number=row["plate_number"],
            body_type=row["body_type"],
            colour=row["colour"],
            make=row["make"],
            model=row["model"],
            rto_city=row["rto_city"],
            synthetic_owner_id=row["synthetic_owner_id"],
        )
    finally:
        conn.close()

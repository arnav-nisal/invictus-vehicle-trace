#!/usr/bin/env python3
"""Synthetic Vehicle Registry & Multi-Camera Sightings Seeder.

Generates and populates SQLite database (data/registry.db) with:
1. Ground-Truth 6 Dummy Benchmark Cases from backend.demo_data (with STOLEN & FLAGGED tags).
2. Deliberate hard-negative lookalikes for each case:
   - OCR character confusion variants (0<->D, 8<->B, 1<->I, 2<->Z, 5<->S)
   - Vehicles sharing the exact same last 4 digits across MH12, MH14, and KA12.
3. Realistic background fleet of synthetic Indian vehicles (total > 1,050).
4. Pre-seeded historical sightings across multi-camera nodes for timeline traceability.
5. 100% synthetic owner names and IDs (strictly privacy preserving, never real personal data).
"""

from __future__ import annotations

import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.demo_data import DUMMY_CASES


# ===========================================================================
# SYNTHETIC CITIZEN & RTO PARAMETERS
# ===========================================================================

SYNTHETIC_FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Aayush",
    "Krishna", "Ishaan", "Shaurya", "Atharva", "Ananya", "Diya", "Saanvi", "Aadhya",
    "Kiara", "Pari", "Fatima", "Zoya", "Rajesh", "Suresh", "Ramesh", "Priya",
    "Sneha", "Pooja", "Sunita", "Amit", "Neha", "Rohit", "Vikram", "Rahul",
    "Deepak", "Manoj", "Sanjay", "Alok", "Sunil", "Anita", "Geeta", "Meena",
    "Rekha", "Kavita", "Shweta", "Aniket", "Tushar", "Harsh", "Pranav", "Tanvi"
]

SYNTHETIC_LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Deshmukh", "Kulkarni", "Patil", "Nair", "Iyer",
    "Subramanian", "Reddy", "Rao", "Singh", "Gupta", "Agarwal", "Joshi", "Mehta",
    "Shah", "Bhat", "Das", "Sen", "Banerjee", "Chatterjee", "Mukherjee", "Bose",
    "Dutta", "Ghosh", "Roy", "Choudhury", "Mishra", "Pandey", "Tiwari", "Shukla",
    "Dubey", "Jha", "Chauhan", "Yadav", "Malhotra", "Kapoor", "Bhardwaj", "Bhatia"
]

RTO_REGIONS = [
    ("MH", "01", "Mumbai South, Maharashtra"),
    ("MH", "02", "Mumbai West, Maharashtra"),
    ("MH", "04", "Thane, Maharashtra"),
    ("MH", "12", "Pune, Maharashtra"),
    ("MH", "14", "Pimpri-Chinchwad, Maharashtra"),
    ("MH", "15", "Nashik, Maharashtra"),
    ("MH", "31", "Nagpur, Maharashtra"),
    ("DL", "01", "Delhi North, Delhi"),
    ("DL", "03", "Delhi Central, Delhi"),
    ("DL", "08", "Delhi Northwest, Delhi"),
    ("DL", "10", "Delhi West, Delhi"),
    ("KA", "01", "Bengaluru Central, Karnataka"),
    ("KA", "03", "Bengaluru East, Karnataka"),
    ("KA", "05", "Bengaluru South, Karnataka"),
    ("KA", "09", "Mysuru, Karnataka"),
    ("KA", "12", "Madikeri (Kodagu), Karnataka"),
    ("KA", "19", "Mangaluru, Karnataka"),
    ("TN", "01", "Chennai Central, Tamil Nadu"),
    ("TN", "07", "Chennai South, Tamil Nadu"),
    ("TN", "09", "Chennai West, Tamil Nadu"),
    ("TN", "37", "Coimbatore, Tamil Nadu"),
    ("TN", "58", "Madurai, Tamil Nadu"),
    ("TS", "07", "Hyderabad Central, Telangana"),
    ("TS", "09", "Hyderabad North, Telangana"),
    ("TS", "10", "Secunderabad, Telangana"),
    ("GJ", "01", "Ahmedabad, Gujarat"),
    ("GJ", "05", "Surat, Gujarat"),
    ("GJ", "06", "Vadodara, Gujarat"),
    ("UP", "16", "Noida, Uttar Pradesh"),
    ("UP", "14", "Ghaziabad, Uttar Pradesh"),
    ("UP", "32", "Lucknow, Uttar Pradesh"),
    ("UP", "78", "Kanpur, Uttar Pradesh"),
    ("HR", "26", "Gurugram, Haryana"),
    ("HR", "51", "Faridabad, Haryana"),
    ("WB", "01", "Kolkata, West Bengal"),
    ("WB", "02", "Kolkata North, West Bengal"),
    ("RJ", "14", "Jaipur, Rajasthan"),
    ("RJ", "19", "Jodhpur, Rajasthan"),
    ("KL", "01", "Thiruvananthapuram, Kerala"),
    ("KL", "07", "Kochi, Kerala"),
    ("MP", "04", "Bhopal, Madhya Pradesh"),
    ("MP", "09", "Indore, Madhya Pradesh"),
]

VEHICLE_CATALOG = {
    "two-wheeler": {
        "makes": [
            ("Honda", ["Activa 6G", "Activa 125", "Shine 125", "Dio 125"]),
            ("TVS", ["Jupiter", "Apache RTR 160", "Raider 125", "Ntorq 125"]),
            ("Bajaj", ["Pulsar 150", "Pulsar NS200", "Platina 110", "Chetak EV"]),
            ("Hero", ["Splendor Plus", "HF Deluxe", "Glamour", "Passion Pro"]),
            ("Suzuki", ["Access 125", "Burgman Street", "Gixxer"]),
            ("Royal Enfield", ["Classic 350", "Hunter 350", "Bullet 350", "Meteor 350"]),
            ("Ather", ["450X", "450S"]),
            ("Ola", ["S1 Pro", "S1 Air"]),
        ],
        "weight": 0.40,
    },
    "car": {
        "makes": [
            ("Maruti Suzuki", ["Swift", "Baleno", "Dzire", "WagonR", "Alto K10", "Celerio", "Ciaz"]),
            ("Hyundai", ["i20", "Grand i10 Nios", "Verna", "Aura"]),
            ("Honda", ["City", "Amaze", "Elevate"]),
            ("Tata", ["Tiago", "Altroz", "Tigor"]),
            ("Volkswagen", ["Virtus", "Taigun", "Polo"]),
            ("Skoda", ["Slavia", "Kushaq"]),
        ],
        "weight": 0.25,
    },
    "suv": {
        "makes": [
            ("Tata", ["Nexon", "Harrier", "Safari", "Punch"]),
            ("Mahindra", ["Scorpio-N", "Scorpio Classic", "XUV700", "Thar", "Bolero", "XUV300"]),
            ("Hyundai", ["Creta", "Venue", "Alcazar"]),
            ("Kia", ["Seltos", "Sonet", "Carens"]),
            ("Maruti Suzuki", ["Brezza", "Grand Vitara", "Fronx"]),
            ("Toyota", ["Fortuner", "Urban Cruiser Hyryder", "Innova Crysta"]),
        ],
        "weight": 0.20,
    },
    "auto-rickshaw": {
        "makes": [
            ("Bajaj", ["Compact 4S", "Maxima Z", "Maxima C", "RE Optima"]),
            ("TVS", ["King Deluxe", "King Duramax", "King Kargo"]),
            ("Piaggio", ["Ape City", "Ape Auto DX", "Ape Xtra LDX"]),
            ("Mahindra", ["Treo", "Alfa"]),
        ],
        "weight": 0.08,
    },
    "truck": {
        "makes": [
            ("Tata", ["407 Gold", "1109 LPT", "Signa 2823.K", "Ultra T.7"]),
            ("Ashok Leyland", ["Dost+", "Bada Dost", "Ecomet 1215", "Boss 1114"]),
            ("Eicher", ["Pro 2049", "Pro 3015", "Pro 6028"]),
            ("BharatBenz", ["1217C", "1617R"]),
        ],
        "weight": 0.04,
    },
    "bus": {
        "makes": [
            ("Tata", ["Starbus Ultra", "Marcopolo", "LP 712"]),
            ("Ashok Leyland", ["Viking", "Oyster", "Lynx Smart"]),
            ("Eicher", ["Skyline Pro", "Starline"]),
            ("Volvo", ["9400 B11R", "9600"]),
        ],
        "weight": 0.03,
    },
}

COLOUR_PALETTE = [
    ("white", 0.35),
    ("silver", 0.20),
    ("grey", 0.15),
    ("black", 0.12),
    ("red", 0.08),
    ("blue", 0.05),
    ("yellow", 0.02),
    ("brown", 0.01),
    ("green", 0.01),
    ("orange", 0.01),
]

SERIES_LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
    "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]


# ===========================================================================
# DATABASE INITIALIZATION
# ===========================================================================

def init_db(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with upgraded schema and optimized indices."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        try:
            db_path.unlink()  # Clean re-seed
        except Exception:
            pass

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 1. Vehicles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            body_type TEXT NOT NULL,
            colour TEXT NOT NULL,
            make TEXT NOT NULL,
            model TEXT NOT NULL,
            rto_city TEXT NOT NULL,
            synthetic_owner_id TEXT NOT NULL,
            owner_name TEXT NOT NULL,
            registered_rto TEXT NOT NULL,
            registration_year INTEGER NOT NULL,
            watchlist_status TEXT NOT NULL DEFAULT 'CLEAR',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles(plate_number);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicles_body_colour ON vehicles(body_type, colour);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicles_make_model ON vehicles(make, model);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicles_watchlist ON vehicles(watchlist_status);")

    # 2. Sightings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sightings (
            sighting_id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            camera_id TEXT NOT NULL DEFAULT 'CAM_01',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            top_matched_plate TEXT NOT NULL,
            confidence REAL NOT NULL,
            crop_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_plate ON sightings(top_matched_plate);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_timestamp ON sightings(timestamp);")

    conn.commit()
    return conn


def generate_synthetic_plate(existing_plates: Set[str]) -> Tuple[str, str, str]:
    """Generate a realistic, unique Indian vehicle registration plate."""
    state, rto_num, rto_city = random.choice(RTO_REGIONS)
    series_len = random.choice([1, 2])
    series = "".join(random.choice(SERIES_LETTERS) for _ in range(series_len))
    num = f"{random.randint(1, 9999):04d}"

    plate = f"{state}{rto_num}{series}{num}"
    while plate in existing_plates:
        series = "".join(random.choice(SERIES_LETTERS) for _ in range(series_len))
        num = f"{random.randint(1, 9999):04d}"
        plate = f"{state}{rto_num}{series}{num}"

    existing_plates.add(plate)
    registered_rto = f"{state}-{rto_num} {rto_city.split(',')[0]} RTO, {rto_city.split(',')[1].strip()}"
    return plate, rto_city, registered_rto


def generate_synthetic_owner(idx: int) -> Tuple[str, str]:
    """Generate 100% synthetic privacy-preserving owner token and name."""
    random_hex = f"{random.randint(0x1000, 0xFFFF):04X}"
    owner_id = f"SYN-IND-{idx:05d}-{random_hex}"
    name = f"{random.choice(SYNTHETIC_FIRST_NAMES)} {random.choice(SYNTHETIC_LAST_NAMES)}"
    return owner_id, name


# ===========================================================================
# SEEDING ROUTINE
# ===========================================================================

def seed_database(db_path: Path, synthetic_target_count: int = 1080):
    """Seed the database with ground-truth test cases, hard negatives, fleet, and sightings."""
    random.seed(42)  # Deterministic repeatability

    conn = init_db(db_path)
    cursor = conn.cursor()

    existing_plates: Set[str] = set()
    records_to_insert: List[Tuple[Any, ...]] = []

    print("=" * 75)
    print("STEP 1: SEEDING 6 GROUND-TRUTH BENCHMARK CASES FROM backend/demo_data.py")
    print("=" * 75)

    gt_plates: List[str] = []
    for case in DUMMY_CASES:
        p = case["plate_number"]
        existing_plates.add(p)
        gt_plates.append(p)
        records_to_insert.append((
            case["plate_number"],
            case["body_type"],
            case["colour"],
            case["make"],
            case["model"],
            case["rto_city"],
            case["synthetic_owner_id"],
            case["owner_name"],
            case["registered_rto"],
            case["registration_year"],
            case["watchlist_status"],
        ))
        print(
            f"  [+] {case['case_id']}: {case['plate_number']} | {case['colour']} {case['body_type']} "
            f"({case['make']} {case['model']}) | Owner: {case['owner_name']} | Status: {case['watchlist_status']}"
        )

    print("\n" + "=" * 75)
    print("STEP 2: GENERATING DELIBERATE HARD-NEGATIVE LOOKALIKES FOR EACH CASE")
    print("=" * 75)

    owner_idx = 10000
    hard_negative_count = 0

    for case in DUMMY_CASES:
        plate = case["plate_number"]
        body_type = case["body_type"]
        colour = case["colour"]
        make = case["make"]
        model = case["model"]
        last_4 = plate[-4:]

        print(f"\n  Generating hard negatives for {case['case_id']} ({plate}, tail={last_4})...")

        # --- A. OCR Character Confusion Lookalikes (0<->D, 8<->B, 1<->I, 2<->Z, 5<->S) ---
        ocr_variants: List[Tuple[str, str]] = []

        # 0 <-> D swap
        if "D" in plate:
            ocr_variants.append((plate.replace("D", "0", 1), "0<->D substitution"))
        if "0" in plate:
            ocr_variants.append((plate.replace("0", "D", 1), "0<->D substitution"))
        if "D" not in plate and "0" not in plate:
            # Create a confusable plate swapping first digit or letter
            ocr_variants.append((plate[:4] + "0" + plate[5:], "0 insertion confusable"))

        # 8 <-> B swap
        if "B" in plate:
            ocr_variants.append((plate.replace("B", "8", 1), "8<->B substitution"))
        if "8" in plate:
            ocr_variants.append((plate.replace("8", "B", 1), "8<->B substitution"))
        if "B" not in plate and "8" not in plate:
            ocr_variants.append((plate[:4] + "B" + plate[5:], "B insertion confusable"))

        # 1 <-> I swap
        if "1" in plate:
            ocr_variants.append((plate.replace("1", "I", 1), "1<->I substitution"))
        if "I" in plate:
            ocr_variants.append((plate.replace("I", "1", 1), "1<->I substitution"))

        # 2 <-> Z swap
        if "2" in plate:
            ocr_variants.append((plate.replace("2", "Z", 1), "2<->Z substitution"))
        if "Z" in plate:
            ocr_variants.append((plate.replace("Z", "2", 1), "2<->Z substitution"))

        # 5 <-> S swap
        if "5" in plate:
            ocr_variants.append((plate.replace("5", "S", 1), "5<->S substitution"))

        for var_plate, reason in ocr_variants:
            if var_plate not in existing_plates:
                existing_plates.add(var_plate)
                owner_idx += 1
                oid, oname = generate_synthetic_owner(owner_idx)
                # Shift colour or model slightly for realistic decoy diversity
                decoy_colour = "silver" if colour == "white" else ("maroon" if colour == "red" else "black")
                records_to_insert.append((
                    var_plate,
                    body_type,
                    decoy_colour,
                    make,
                    model,
                    case["rto_city"],
                    oid,
                    oname,
                    case["registered_rto"],
                    case["registration_year"],
                    "CLEAR",
                ))
                hard_negative_count += 1
                print(f"    * OCR Lookalike ({reason}): {var_plate} ({decoy_colour} {body_type})")

        # --- B. Cross-Jurisdiction Last-4 Digits Lookalikes across MH12, MH14, and KA12 ---
        jurisdiction_targets = [
            ("MH", "12", "Pune, Maharashtra", "MH-12 Pune RTO, Maharashtra"),
            ("MH", "14", "Pimpri-Chinchwad, Maharashtra", "MH-14 Pimpri-Chinchwad RTO, Maharashtra"),
            ("KA", "12", "Madikeri (Kodagu), Karnataka", "KA-12 Madikeri RTO, Karnataka"),
        ]

        for st, code, city, rto_name in jurisdiction_targets:
            # Generate 1 or 2 distinct series for each jurisdiction with the exact last 4 digits
            series_candidates = ["AB", "CD", "EF", "DE", "AZ", "CB", "MJ"]
            for s in series_candidates:
                cand_plate = f"{st}{code}{s}{last_4}"
                if cand_plate not in existing_plates:
                    existing_plates.add(cand_plate)
                    owner_idx += 1
                    oid, oname = generate_synthetic_owner(owner_idx)
                    year = random.randint(2017, 2023)
                    records_to_insert.append((
                        cand_plate,
                        body_type,
                        colour,
                        make,
                        model,
                        city,
                        oid,
                        oname,
                        rto_name,
                        year,
                        "CLEAR",
                    ))
                    hard_negative_count += 1
                    print(f"    * Tail Lookalike ({st}{code}): {cand_plate} in {city}")
                    break  # 1 per target jurisdiction is sufficient

    print(f"\n[+] Total hard-negative lookalikes generated: {hard_negative_count}")

    # --- 3. Background Synthetic Indian Fleet (to reach >= 1,050 total) ---
    print("\n" + "=" * 75)
    print(f"STEP 3: GENERATING BACKGROUND FLEET (TARGETING >= 1,050 TOTAL VEHICLES)")
    print("=" * 75)

    body_types = list(VEHICLE_CATALOG.keys())
    body_type_weights = [VEHICLE_CATALOG[bt]["weight"] for bt in body_types]
    colours = [c[0] for c in COLOUR_PALETTE]
    colour_weights = [c[1] for c in COLOUR_PALETTE]

    current_count = len(records_to_insert)
    needed = max(0, synthetic_target_count - current_count)
    print(f"Current records: {current_count}. Generating {needed} additional synthetic vehicles...")

    for i in range(1, needed + 1):
        chosen_body = random.choices(body_types, weights=body_type_weights, k=1)[0]
        makes_list = VEHICLE_CATALOG[chosen_body]["makes"]
        chosen_make, models_list = random.choice(makes_list)
        chosen_model = random.choice(models_list)
        chosen_colour = random.choices(colours, weights=colour_weights, k=1)[0]

        plate, rto_city, registered_rto = generate_synthetic_plate(existing_plates)
        owner_idx += 1
        owner_id, owner_name = generate_synthetic_owner(owner_idx)
        year = random.randint(2015, 2024)

        # Distribute realistic surveillance watchlist tags (98% CLEAR, 1% STOLEN, 1% FLAGGED)
        tag_roll = random.random()
        if tag_roll < 0.01:
            watchlist_status = "STOLEN"
        elif tag_roll < 0.02:
            watchlist_status = "FLAGGED"
        else:
            watchlist_status = "CLEAR"

        records_to_insert.append((
            plate,
            chosen_body,
            chosen_colour,
            chosen_make,
            chosen_model,
            rto_city,
            owner_id,
            owner_name,
            registered_rto,
            year,
            watchlist_status,
        ))

    # Bulk insert vehicles
    cursor.executemany("""
        INSERT INTO vehicles (
            plate_number, body_type, colour, make, model, rto_city,
            synthetic_owner_id, owner_name, registered_rto, registration_year, watchlist_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, records_to_insert)

    conn.commit()

    # --- 4. Pre-seed Historical Sightings for Primary Demo Plate ---
    print("\n" + "=" * 75)
    print("STEP 4: PRE-SEEDING MULTI-CAMERA HISTORICAL SIGHTINGS")
    print("=" * 75)

    primary_plate = "MH12DE1432"
    historical_sightings = [
        (
            101,
            "CAM_01",
            "2026-11-23 08:30:15",
            primary_plate,
            0.985,
            "outputs/crops/track_001_frame_0042_vehicle.jpg",
        ),
        (
            142,
            "CAM_04",
            "2026-11-23 08:42:50",
            primary_plate,
            0.972,
            "outputs/crops/track_001_frame_0048_vehicle.jpg",
        ),
        (
            189,
            "CAM_07",
            "2026-11-23 09:05:22",
            primary_plate,
            0.991,
            "outputs/crops/track_001_frame_0042_vehicle.jpg",
        ),
        # Also seed 1 sighting for our STOLEN SUV (DL03CB8051)
        (
            204,
            "CAM_02",
            "2026-11-23 09:12:00",
            "DL03CB8051",
            0.988,
            "outputs/crops/track_002_frame_0088_vehicle.jpg",
        ),
    ]

    cursor.executemany("""
        INSERT INTO sightings (track_id, camera_id, timestamp, top_matched_plate, confidence, crop_path)
        VALUES (?, ?, ?, ?, ?, ?);
    """, historical_sightings)

    conn.commit()

    # Verify counts
    cursor.execute("SELECT COUNT(*) FROM vehicles;")
    total_vehicles = cursor.fetchone()[0]

    cursor.execute("SELECT watchlist_status, COUNT(*) FROM vehicles GROUP BY watchlist_status;")
    watchlist_counts = dict(cursor.fetchall())

    cursor.execute("SELECT COUNT(*) FROM sightings;")
    total_sightings = cursor.fetchone()[0]

    conn.close()

    print(f"\n[SUCCESS] Seeded {total_vehicles} total vehicles into '{db_path}'.")
    print(f"  - Ground-truth target cases:     {len(DUMMY_CASES)}")
    print(f"  - Hard near-miss lookalikes:     {hard_negative_count}")
    print(f"  - Total vehicle fleet in DB:     {total_vehicles}")
    print(f"  - Watchlist breakdown:           {watchlist_counts}")
    print(f"  - Pre-seeded sightings in DB:    {total_sightings}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    db_file = base_dir / "data" / "registry.db"
    seed_database(db_file, synthetic_target_count=1080)

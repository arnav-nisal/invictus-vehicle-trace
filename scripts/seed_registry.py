#!/usr/bin/env python3
"""Synthetic Vehicle Registry & Multi-Camera Sightings Seeder.

Generates and populates SQLite database (data/registry.db) with:
1. Ground-Truth 6 Dummy Benchmark Cases from backend.demo_data (with STOLEN & FLAGGED tags).
2. 23 Ground-Truth Vehicles from test clips (file_01.mp4 to file_07.mp4 and activa_signal.mp4).
3. Deliberate hard-negative lookalikes for benchmark cases and target test clip vehicles:
   - OCR character confusion variants (0<->D, 8<->B, 5<->S, 6<->5, N<->M, 1<->I, 2<->Z)
   - Cross-RTO vehicles sharing the exact same last 4 digits.
4. Realistic background fleet of synthetic Indian vehicles (total >= 1,050).
5. Pre-seeded historical sightings across multi-camera nodes for timeline traceability.
6. 100% synthetic owner names and IDs (strictly privacy preserving, never real personal data).
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
# 23 GROUND-TRUTH TEST CLIP VEHICLES
# ===========================================================================

GROUND_TRUTH_VEHICLES: List[Dict[str, Any]] = [
    {
        "plate_number": "MH46BA3651",
        "body_type": "car",
        "colour": "grey",
        "make": "Honda",
        "model": "City",
        "rto_city": "Panvel (Navi Mumbai), Maharashtra",
        "registered_rto": "MH-46 Panvel RTO, Maharashtra",
        "registration_year": 2021,
        "owner_name": "Aditya Deshmukh",
        "synthetic_owner_id": "SYN-IND-GT01-3651",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH01DR3708",
        "body_type": "car",
        "colour": "white",
        "make": "Maruti Suzuki",
        "model": "Eeco",
        "rto_city": "Mumbai South, Maharashtra",
        "registered_rto": "MH-01 Tardeo RTO, Maharashtra",
        "registration_year": 2020,
        "owner_name": "Rajesh Sharma",
        "synthetic_owner_id": "SYN-IND-GT02-3708",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH43BE6764",
        "body_type": "suv",
        "colour": "red",
        "make": "Maruti Suzuki",
        "model": "Vitara Brezza",
        "rto_city": "Navi Mumbai (Vashi), Maharashtra",
        "registered_rto": "MH-43 Vashi RTO, Maharashtra",
        "registration_year": 2019,
        "owner_name": "Suresh Patil",
        "synthetic_owner_id": "SYN-IND-GT03-6764",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH01DB8063",
        "body_type": "suv",
        "colour": "silver",
        "make": "Honda",
        "model": "WR-V",
        "rto_city": "Mumbai South, Maharashtra",
        "registered_rto": "MH-01 Tardeo RTO, Maharashtra",
        "registration_year": 2020,
        "owner_name": "Pooja Verma",
        "synthetic_owner_id": "SYN-IND-GT04-8063",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH02BQ9421",
        "body_type": "car",
        "colour": "black",
        "make": "Hyundai",
        "model": "Santro",
        "rto_city": "Mumbai West, Maharashtra",
        "registered_rto": "MH-02 Andheri RTO, Maharashtra",
        "registration_year": 2018,
        "owner_name": "Kavita Kulkarni",
        "synthetic_owner_id": "SYN-IND-GT05-9421",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH47BL9024",
        "body_type": "car",
        "colour": "white",
        "make": "Maruti Suzuki",
        "model": "WagonR",
        "rto_city": "Mumbai North (Borivali), Maharashtra",
        "registered_rto": "MH-47 Borivali RTO, Maharashtra",
        "registration_year": 2021,
        "owner_name": "Amit Deshmukh",
        "synthetic_owner_id": "SYN-IND-GT06-9024",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "UP15BE5580",
        "body_type": "two-wheeler",
        "colour": "white",
        "make": "Honda",
        "model": "Activa",
        "rto_city": "Meerut, Uttar Pradesh",
        "registered_rto": "UP-15 Meerut RTO, Uttar Pradesh",
        "registration_year": 2022,
        "owner_name": "Deepak Chauhan",
        "synthetic_owner_id": "SYN-IND-GT07-5580",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH01BG7184",
        "body_type": "car",
        "colour": "silver",
        "make": "Honda",
        "model": "Amaze",
        "rto_city": "Mumbai South, Maharashtra",
        "registered_rto": "MH-01 Tardeo RTO, Maharashtra",
        "registration_year": 2021,
        "owner_name": "Sunil Joshi",
        "synthetic_owner_id": "SYN-IND-GT08-7184",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH01CT4829",
        "body_type": "car",
        "colour": "black",
        "make": "Hyundai",
        "model": "Santro",
        "rto_city": "Mumbai South, Maharashtra",
        "registered_rto": "MH-01 Tardeo RTO, Maharashtra",
        "registration_year": 2017,
        "owner_name": "Manoj Bhatia",
        "synthetic_owner_id": "SYN-IND-GT09-4829",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH03DN0759",
        "body_type": "auto-rickshaw",
        "colour": "blue",
        "make": "Piaggio",
        "model": "Ape",
        "rto_city": "Mumbai East (Wadala), Maharashtra",
        "registered_rto": "MH-03 Wadala RTO, Maharashtra",
        "registration_year": 2020,
        "owner_name": "Alok Yadav",
        "synthetic_owner_id": "SYN-IND-GT10-0759",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH47AN6735",
        "body_type": "suv",
        "colour": "white",
        "make": "Kia",
        "model": "Seltos",
        "rto_city": "Mumbai North (Borivali), Maharashtra",
        "registered_rto": "MH-47 Borivali RTO, Maharashtra",
        "registration_year": 2022,
        "owner_name": "Neha Shah",
        "synthetic_owner_id": "SYN-IND-GT11-6735",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH04LM3540",
        "body_type": "suv",
        "colour": "white",
        "make": "Hyundai",
        "model": "Creta",
        "rto_city": "Thane, Maharashtra",
        "registered_rto": "MH-04 Thane RTO, Maharashtra",
        "registration_year": 2023,
        "owner_name": "Sanjay Mehta",
        "synthetic_owner_id": "SYN-IND-GT12-3540",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH01BB6376",
        "body_type": "suv",
        "colour": "white",
        "make": "Toyota",
        "model": "Fortuner",
        "rto_city": "Mumbai South, Maharashtra",
        "registered_rto": "MH-01 Tardeo RTO, Maharashtra",
        "registration_year": 2021,
        "owner_name": "Vihaan Agarwal",
        "synthetic_owner_id": "SYN-IND-GT13-6376",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "GJ15CL8629",
        "body_type": "car",
        "colour": "white",
        "make": "Maruti Suzuki",
        "model": "Baleno",
        "rto_city": "Valsad, Gujarat",
        "registered_rto": "GJ-15 Valsad RTO, Gujarat",
        "registration_year": 2020,
        "owner_name": "Tanvi Patel",
        "synthetic_owner_id": "SYN-IND-GT14-8629",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH01CV4352",
        "body_type": "truck",
        "colour": "white",
        "make": "Tata",
        "model": "Ace",
        "rto_city": "Mumbai South, Maharashtra",
        "registered_rto": "MH-01 Tardeo RTO, Maharashtra",
        "registration_year": 2019,
        "owner_name": "Ramesh Gupta",
        "synthetic_owner_id": "SYN-IND-GT15-4352",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH01CL3421",
        "body_type": "suv",
        "colour": "grey",
        "make": "Toyota",
        "model": "Innova Crysta",
        "rto_city": "Mumbai South, Maharashtra",
        "registered_rto": "MH-01 Tardeo RTO, Maharashtra",
        "registration_year": 2021,
        "owner_name": "Pranav Tiwari",
        "synthetic_owner_id": "SYN-IND-GT16-3421",
        "watchlist_status": "STOLEN",
    },
    {
        "plate_number": "MH04HN1518",
        "body_type": "suv",
        "colour": "red",
        "make": "Maruti Suzuki",
        "model": "Vitara Brezza",
        "rto_city": "Thane, Maharashtra",
        "registered_rto": "MH-04 Thane RTO, Maharashtra",
        "registration_year": 2020,
        "owner_name": "Aniket Rao",
        "synthetic_owner_id": "SYN-IND-GT17-1518",
        "watchlist_status": "FLAGGED",
    },
    {
        "plate_number": "MH12DE1652",
        "body_type": "car",
        "colour": "blue",
        "make": "Maruti Suzuki",
        "model": "Swift",
        "rto_city": "Pune, Maharashtra",
        "registered_rto": "MH-12 Pune RTO, Maharashtra",
        "registration_year": 2022,
        "owner_name": "Ishaan Deshmukh",
        "synthetic_owner_id": "SYN-IND-GT18-1652",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH04KU7812",
        "body_type": "truck",
        "colour": "red",
        "make": "Eicher",
        "model": "Pro 2049",
        "rto_city": "Thane, Maharashtra",
        "registered_rto": "MH-04 Thane RTO, Maharashtra",
        "registration_year": 2018,
        "owner_name": "Tushar Dubey",
        "synthetic_owner_id": "SYN-IND-GT19-7812",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH02EC5190",
        "body_type": "two-wheeler",
        "colour": "white",
        "make": "Suzuki",
        "model": "Access 125",
        "rto_city": "Mumbai West, Maharashtra",
        "registered_rto": "MH-02 Andheri RTO, Maharashtra",
        "registration_year": 2022,
        "owner_name": "Diya Bose",
        "synthetic_owner_id": "SYN-IND-GT20-5190",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH02FG4812",
        "body_type": "two-wheeler",
        "colour": "grey",
        "make": "Suzuki",
        "model": "Burgman Street",
        "rto_city": "Mumbai West, Maharashtra",
        "registered_rto": "MH-02 Andheri RTO, Maharashtra",
        "registration_year": 2023,
        "owner_name": "Shweta Kulkarni",
        "synthetic_owner_id": "SYN-IND-GT21-4812",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH12JK4410",
        "body_type": "car",
        "colour": "red",
        "make": "Honda",
        "model": "City",
        "rto_city": "Pune, Maharashtra",
        "registered_rto": "MH-12 Pune RTO, Maharashtra",
        "registration_year": 2020,
        "owner_name": "Atharva Joshi",
        "synthetic_owner_id": "SYN-IND-GT22-4410",
        "watchlist_status": "CLEAR",
    },
    {
        "plate_number": "MH12RT3095",
        "body_type": "two-wheeler",
        "colour": "black",
        "make": "Hero",
        "model": "Splendor",
        "rto_city": "Pune, Maharashtra",
        "registered_rto": "MH-12 Pune RTO, Maharashtra",
        "registration_year": 2019,
        "owner_name": "Harsh Bhardwaj",
        "synthetic_owner_id": "SYN-IND-GT23-3095",
        "watchlist_status": "CLEAR",
    },
]

TARGET_LOOKALIKE_PLATES: Set[str] = {
    "MH43BE6764",
    "MH01DB8063",
    "MH02BQ9421",
    "UP15BE5580",
    "MH01BG7184",
    "MH46BA3651",
    "MH47AN6735",
    "MH01BB6376",
    "GJ15CL8629",
}


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
    ("MH", "03", "Mumbai East (Wadala), Maharashtra"),
    ("MH", "04", "Thane, Maharashtra"),
    ("MH", "12", "Pune, Maharashtra"),
    ("MH", "14", "Pimpri-Chinchwad, Maharashtra"),
    ("MH", "15", "Nashik, Maharashtra"),
    ("MH", "31", "Nagpur, Maharashtra"),
    ("MH", "43", "Navi Mumbai (Vashi), Maharashtra"),
    ("MH", "46", "Panvel (Navi Mumbai), Maharashtra"),
    ("MH", "47", "Mumbai North (Borivali), Maharashtra"),
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
    ("GJ", "15", "Valsad, Gujarat"),
    ("UP", "14", "Ghaziabad, Uttar Pradesh"),
    ("UP", "15", "Meerut, Uttar Pradesh"),
    ("UP", "16", "Noida, Uttar Pradesh"),
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
    city_part = rto_city.split(",")[0].strip()
    state_part = rto_city.split(",")[1].strip() if "," in rto_city else ""
    registered_rto = f"{state}-{rto_num} {city_part} RTO, {state_part}".strip()
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

    # -----------------------------------------------------------------------
    # STEP 1A: 6 Benchmark Cases from backend.demo_data
    # -----------------------------------------------------------------------
    print("=" * 75)
    print("STEP 1A: SEEDING 6 GROUND-TRUTH BENCHMARK CASES FROM backend/demo_data.py")
    print("=" * 75)

    for case in DUMMY_CASES:
        p = case["plate_number"]
        existing_plates.add(p)
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

    # -----------------------------------------------------------------------
    # STEP 1B: 23 Ground-Truth Test Clip Vehicles
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 1B: SEEDING 23 GROUND-TRUTH VEHICLES FROM 7 TEST CLIPS")
    print("=" * 75)

    for idx, v in enumerate(GROUND_TRUTH_VEHICLES, start=1):
        p = v["plate_number"]
        if p not in existing_plates:
            existing_plates.add(p)
            records_to_insert.append((
                v["plate_number"],
                v["body_type"],
                v["colour"],
                v["make"],
                v["model"],
                v["rto_city"],
                v["synthetic_owner_id"],
                v["owner_name"],
                v["registered_rto"],
                v["registration_year"],
                v["watchlist_status"],
            ))
            print(
                f"  [+] GT-{idx:02d}: {v['plate_number']} | {v['colour']} {v['body_type']} "
                f"({v['make']} {v['model']}) | RTO: {v['registered_rto']} | Status: {v['watchlist_status']}"
            )

    # -----------------------------------------------------------------------
    # STEP 2A: Deliberate Hard Negatives for 6 Benchmark Cases
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 2A: GENERATING DELIBERATE HARD-NEGATIVE LOOKALIKES FOR 6 BENCHMARK CASES")
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
                    break

    # -----------------------------------------------------------------------
    # STEP 2B: Deliberate Hard Negatives for 9 Target Ground-Truth Plates
    # OCR confusion lookalikes (0<->D, 8<->B, 5<->S, 6<->5, N<->M) & cross-RTO
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 2B: GENERATING TARGET HARD NEGATIVES (OCR CONFUSION & CROSS-RTO) FOR 9 GT PLATES")
    print("=" * 75)

    cross_rto_pool = [
        ("MH", "12", "Pune, Maharashtra", "MH-12 Pune RTO, Maharashtra"),
        ("MH", "01", "Mumbai South, Maharashtra", "MH-01 Tardeo RTO, Maharashtra"),
        ("MH", "04", "Thane, Maharashtra", "MH-04 Thane RTO, Maharashtra"),
        ("MH", "14", "Pimpri-Chinchwad, Maharashtra", "MH-14 Pimpri-Chinchwad RTO, Maharashtra"),
        ("MH", "47", "Mumbai North, Maharashtra", "MH-47 Borivali RTO, Maharashtra"),
        ("MH", "43", "Navi Mumbai, Maharashtra", "MH-43 Vashi RTO, Maharashtra"),
        ("KA", "12", "Madikeri (Kodagu), Karnataka", "KA-12 Madikeri RTO, Karnataka"),
        ("GJ", "15", "Valsad, Gujarat", "GJ-15 Valsad RTO, Gujarat"),
        ("UP", "15", "Meerut, Uttar Pradesh", "UP-15 Meerut RTO, Uttar Pradesh"),
    ]

    target_gt_records = [v for v in GROUND_TRUTH_VEHICLES if v["plate_number"] in TARGET_LOOKALIKE_PLATES]

    for gt in target_gt_records:
        plate = gt["plate_number"]
        body_type = gt["body_type"]
        colour = gt["colour"]
        make = gt["make"]
        model = gt["model"]
        last_4 = plate[-4:]
        series = plate[4:-4]
        prefix = plate[:4]

        print(f"\n  Generating competing lookalikes for {plate} ({colour} {body_type} {make} {model})...")

        # --- A. OCR Character Confusion Lookalikes (0<->D, 8<->B, 5<->S, 6<->5, N<->M) ---
        target_ocr_variants: List[Tuple[str, str]] = []

        for i, ch in enumerate(plate):
            # 0 <-> D
            if ch == "0":
                target_ocr_variants.append((plate[:i] + "D" + plate[i + 1:], "0<->D substitution"))
            elif ch == "D":
                target_ocr_variants.append((plate[:i] + "0" + plate[i + 1:], "0<->D substitution"))
            # 8 <-> B
            elif ch == "8":
                target_ocr_variants.append((plate[:i] + "B" + plate[i + 1:], "8<->B substitution"))
            elif ch == "B":
                target_ocr_variants.append((plate[:i] + "8" + plate[i + 1:], "8<->B substitution"))
            # 5 <-> S and 6 <-> 5
            elif ch == "5":
                target_ocr_variants.append((plate[:i] + "S" + plate[i + 1:], "5<->S substitution"))
                target_ocr_variants.append((plate[:i] + "6" + plate[i + 1:], "6<->5 substitution"))
            elif ch == "S":
                target_ocr_variants.append((plate[:i] + "5" + plate[i + 1:], "5<->S substitution"))
            elif ch == "6":
                target_ocr_variants.append((plate[:i] + "5" + plate[i + 1:], "6<->5 substitution"))
            # N <-> M
            elif ch == "N":
                target_ocr_variants.append((plate[:i] + "M" + plate[i + 1:], "N<->M substitution"))
            elif ch == "M":
                target_ocr_variants.append((plate[:i] + "N" + plate[i + 1:], "N<->M substitution"))
            # 1 <-> I
            elif ch == "1":
                target_ocr_variants.append((plate[:i] + "I" + plate[i + 1:], "1<->I substitution"))
            elif ch == "I":
                target_ocr_variants.append((plate[:i] + "1" + plate[i + 1:], "1<->I substitution"))
            # 2 <-> Z
            elif ch == "2":
                target_ocr_variants.append((plate[:i] + "Z" + plate[i + 1:], "2<->Z substitution"))
            elif ch == "Z":
                target_ocr_variants.append((plate[:i] + "2" + plate[i + 1:], "2<->Z substitution"))

        for var_plate, reason in target_ocr_variants:
            if var_plate not in existing_plates:
                existing_plates.add(var_plate)
                owner_idx += 1
                oid, oname = generate_synthetic_owner(owner_idx)
                # Keep matching attributes so candidate competes realistically in top 3
                records_to_insert.append((
                    var_plate,
                    body_type,
                    colour,
                    make,
                    model,
                    gt["rto_city"],
                    oid,
                    oname,
                    gt["registered_rto"],
                    gt["registration_year"],
                    "CLEAR",
                ))
                hard_negative_count += 1
                print(f"    * OCR Lookalike ({reason}): {var_plate} ({colour} {body_type})")

        # --- B. Cross-RTO Last-4 Digits Lookalikes ---
        tail_lookalike_added = 0
        for st, code, city, rto_name in cross_rto_pool:
            if f"{st}{code}" == prefix:
                continue
            cand_plate = None
            for s in [series, "AB", "CD", "EF"]:
                trial = f"{st}{code}{s}{last_4}"
                if trial not in existing_plates:
                    cand_plate = trial
                    break
            if cand_plate:
                existing_plates.add(cand_plate)
                owner_idx += 1
                oid, oname = generate_synthetic_owner(owner_idx)
                year = random.randint(2018, 2023)
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
                tail_lookalike_added += 1
                print(f"    * Cross-RTO Lookalike ({st}{code}): {cand_plate} in {city}")
                if tail_lookalike_added >= 3:
                    break

    print(f"\n[+] Total hard-negative lookalikes generated: {hard_negative_count}")

    # -----------------------------------------------------------------------
    # STEP 3: Background Synthetic Indian Fleet (to reach >= 1,050 total)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 3: GENERATING BACKGROUND FLEET (TARGETING >= 1,050 TOTAL VEHICLES)")
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

    # -----------------------------------------------------------------------
    # STEP 4: Pre-seed Multi-Camera Historical Sightings
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STEP 4: PRE-SEEDING MULTI-CAMERA HISTORICAL SIGHTINGS")
    print("=" * 75)

    primary_plate = "MH12DE1432"
    historical_sightings = [
        # Primary calibration benchmark plate (MH12DE1432)
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
        # Seed 1 sighting for benchmark STOLEN SUV (DL03CB8051)
        (
            204,
            "CAM_02",
            "2026-11-23 09:12:00",
            "DL03CB8051",
            0.988,
            "outputs/crops/track_002_frame_0088_vehicle.jpg",
        ),

        # Ground-truth test clip vehicle sightings:
        # 1. MH46BA3651 (3 chronological sightings across CAM_01, CAM_04, CAM_07)
        (
            301,
            "CAM_01",
            "2026-11-23 08:15:10",
            "MH46BA3651",
            0.982,
            "outputs/crops/mh46ba3651_cam01.jpg",
        ),
        (
            302,
            "CAM_04",
            "2026-11-23 08:32:45",
            "MH46BA3651",
            0.975,
            "outputs/crops/mh46ba3651_cam04.jpg",
        ),
        (
            303,
            "CAM_07",
            "2026-11-23 08:52:30",
            "MH46BA3651",
            0.989,
            "outputs/crops/mh46ba3651_cam07.jpg",
        ),

        # 2. MH01DR3708 (2 sightings across CAM_01 and CAM_04)
        (
            304,
            "CAM_01",
            "2026-11-23 08:20:15",
            "MH01DR3708",
            0.978,
            "outputs/crops/mh01dr3708_cam01.jpg",
        ),
        (
            305,
            "CAM_04",
            "2026-11-23 08:38:20",
            "MH01DR3708",
            0.984,
            "outputs/crops/mh01dr3708_cam04.jpg",
        ),

        # 7. UP15BE5580 (2 sightings across CAM_02 and CAM_05)
        (
            306,
            "CAM_02",
            "2026-11-23 09:05:10",
            "UP15BE5580",
            0.982,
            "outputs/crops/up15be5580_cam02.jpg",
        ),
        (
            307,
            "CAM_05",
            "2026-11-23 09:22:40",
            "UP15BE5580",
            0.976,
            "outputs/crops/up15be5580_cam05.jpg",
        ),

        # 8. MH01BG7184 (2 sightings across CAM_01 and CAM_03)
        (
            308,
            "CAM_01",
            "2026-11-23 08:45:00",
            "MH01BG7184",
            0.981,
            "outputs/crops/mh01bg7184_cam01.jpg",
        ),
        (
            309,
            "CAM_03",
            "2026-11-23 09:02:15",
            "MH01BG7184",
            0.977,
            "outputs/crops/mh01bg7184_cam03.jpg",
        ),

        # 12. MH04LM3540 (2 sightings across CAM_03 and CAM_06)
        (
            310,
            "CAM_03",
            "2026-11-23 09:15:30",
            "MH04LM3540",
            0.987,
            "outputs/crops/mh04lm3540_cam03.jpg",
        ),
        (
            311,
            "CAM_06",
            "2026-11-23 09:35:50",
            "MH04LM3540",
            0.982,
            "outputs/crops/mh04lm3540_cam06.jpg",
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
    print(f"  - Ground-truth target cases:     {len(DUMMY_CASES) + len(GROUND_TRUTH_VEHICLES)}")
    print(f"  - Hard near-miss lookalikes:     {hard_negative_count}")
    print(f"  - Total vehicle fleet in DB:     {total_vehicles}")
    print(f"  - Watchlist breakdown:           {watchlist_counts}")
    print(f"  - Pre-seeded sightings in DB:    {total_sightings}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    db_file = base_dir / "data" / "registry.db"
    seed_database(db_file, synthetic_target_count=1080)

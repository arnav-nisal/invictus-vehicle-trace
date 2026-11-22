#!/usr/bin/env python3
"""Synthetic Vehicle Registry Seeder.

Generates and populates SQLite database (data/registry.db) with:
1. Ground-Truth Test Vehicles and Near-Miss Hard Decoys (configurable at top).
2. 1,000 realistic synthetic Indian vehicles across diverse states, body types, and colours.
3. 100% synthetic owner IDs (never real names).
"""

import os
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Any


# ===========================================================================
# GROUND-TRUTH TEST VEHICLES & NEAR-MISS HARD DECOYS
# (Easily update these entries to match ground-truth plates from test clips)
# ===========================================================================

GROUND_TRUTH_TEST_VEHICLES: List[Dict[str, Any]] = [
    {
        "test_case": "GT-1 (White Scooter / Two-Wheeler - clip target)",
        "ground_truth": {
            "plate_number": "MH12DE1432",
            "body_type": "two-wheeler",
            "colour": "white",
            "make": "Honda",
            "model": "Activa 6G",
            "rto_city": "Pune, Maharashtra",
            "synthetic_owner_id": "SYN-OWN-1001",
        },
        "decoys": [
            {
                "label": "Decoy 1A: Character confusion (1 vs I)",
                "plate_number": "MH12DEI432",
                "body_type": "two-wheeler",
                "colour": "white",
                "make": "Honda",
                "model": "Activa 125",
                "rto_city": "Pune, Maharashtra",
                "synthetic_owner_id": "SYN-OWN-1002",
            },
            {
                "label": "Decoy 1B: Character confusion (D vs 0)",
                "plate_number": "MH120E1432",
                "body_type": "two-wheeler",
                "colour": "white",
                "make": "TVS",
                "model": "Jupiter",
                "rto_city": "Pune, Maharashtra",
                "synthetic_owner_id": "SYN-OWN-1003",
            },
            {
                "label": "Decoy 1C: Character confusion (2 vs Z) + Colour shift",
                "plate_number": "MH12DE143Z",
                "body_type": "two-wheeler",
                "colour": "silver",
                "make": "Suzuki",
                "model": "Access 125",
                "rto_city": "Pune, Maharashtra",
                "synthetic_owner_id": "SYN-OWN-1004",
            },
        ],
    },
    {
        "test_case": "GT-2 (Red Compact SUV)",
        "ground_truth": {
            "plate_number": "DL03CB8051",
            "body_type": "suv",
            "colour": "red",
            "make": "Tata",
            "model": "Nexon",
            "rto_city": "Delhi Central, Delhi",
            "synthetic_owner_id": "SYN-OWN-1005",
        },
        "decoys": [
            {
                "label": "Decoy 2A: Character confusion (B vs 8)",
                "plate_number": "DL03C88051",
                "body_type": "suv",
                "colour": "red",
                "make": "Tata",
                "model": "Nexon",
                "rto_city": "Delhi Central, Delhi",
                "synthetic_owner_id": "SYN-OWN-1006",
            },
            {
                "label": "Decoy 2B: Character confusion (0 vs O)",
                "plate_number": "DL03CBO051",
                "body_type": "suv",
                "colour": "red",
                "make": "Mahindra",
                "model": "XUV300",
                "rto_city": "Delhi Central, Delhi",
                "synthetic_owner_id": "SYN-OWN-1007",
            },
            {
                "label": "Decoy 2C: Character confusion (5 vs S) + Body shift",
                "plate_number": "DL03CB80S1",
                "body_type": "car",
                "colour": "maroon",
                "make": "Hyundai",
                "model": "Venue",
                "rto_city": "Delhi Central, Delhi",
                "synthetic_owner_id": "SYN-OWN-1008",
            },
        ],
    },
    {
        "test_case": "GT-3 (Silver Sedan / Car)",
        "ground_truth": {
            "plate_number": "KA01MJ4096",
            "body_type": "car",
            "colour": "silver",
            "make": "Maruti Suzuki",
            "model": "Dzire",
            "rto_city": "Bengaluru Central, Karnataka",
            "synthetic_owner_id": "SYN-OWN-1009",
        },
        "decoys": [
            {
                "label": "Decoy 3A: Character confusion (0 vs O)",
                "plate_number": "KA01MJO096",
                "body_type": "car",
                "colour": "silver",
                "make": "Maruti Suzuki",
                "model": "Ciaz",
                "rto_city": "Bengaluru Central, Karnataka",
                "synthetic_owner_id": "SYN-OWN-1010",
            },
            {
                "label": "Decoy 3B: Character confusion (6 vs G) + Colour shift",
                "plate_number": "KA01MJ409G",
                "body_type": "car",
                "colour": "grey",
                "make": "Honda",
                "model": "Amaze",
                "rto_city": "Bengaluru Central, Karnataka",
                "synthetic_owner_id": "SYN-OWN-1011",
            },
            {
                "label": "Decoy 3C: Character confusion (1 vs I) + Colour shift",
                "plate_number": "KA0IMJ4096",
                "body_type": "car",
                "colour": "white",
                "make": "Hyundai",
                "model": "Aura",
                "rto_city": "Bengaluru Central, Karnataka",
                "synthetic_owner_id": "SYN-OWN-1012",
            },
        ],
    },
    {
        "test_case": "GT-4 (Yellow Auto-Rickshaw)",
        "ground_truth": {
            "plate_number": "MH02AZ5820",
            "body_type": "auto-rickshaw",
            "colour": "yellow",
            "make": "Bajaj",
            "model": "Compact 4S",
            "rto_city": "Mumbai West, Maharashtra",
            "synthetic_owner_id": "SYN-OWN-1013",
        },
        "decoys": [
            {
                "label": "Decoy 4A: Character confusion (8 vs B)",
                "plate_number": "MH02AZ5B20",
                "body_type": "auto-rickshaw",
                "colour": "yellow",
                "make": "Bajaj",
                "model": "Maxima Z",
                "rto_city": "Mumbai West, Maharashtra",
                "synthetic_owner_id": "SYN-OWN-1014",
            },
            {
                "label": "Decoy 4B: Character confusion (5 vs S) + Dual tone",
                "plate_number": "MH02AZS820",
                "body_type": "auto-rickshaw",
                "colour": "black-yellow",
                "make": "Piaggio",
                "model": "Ape City",
                "rto_city": "Mumbai West, Maharashtra",
                "synthetic_owner_id": "SYN-OWN-1015",
            },
            {
                "label": "Decoy 4C: Character confusion (0 vs D) + Colour shift",
                "plate_number": "MH02AZ582D",
                "body_type": "auto-rickshaw",
                "colour": "green",
                "make": "Bajaj",
                "model": "RE EV",
                "rto_city": "Mumbai West, Maharashtra",
                "synthetic_owner_id": "SYN-OWN-1016",
            },
        ],
    },
    {
        "test_case": "GT-5 (Blue Medium Commercial Truck)",
        "ground_truth": {
            "plate_number": "TN09BY7104",
            "body_type": "truck",
            "colour": "blue",
            "make": "Tata",
            "model": "407 Gold",
            "rto_city": "Chennai South, Tamil Nadu",
            "synthetic_owner_id": "SYN-OWN-1017",
        },
        "decoys": [
            {
                "label": "Decoy 5A: Character confusion (B vs 8)",
                "plate_number": "TN098Y7104",
                "body_type": "truck",
                "colour": "blue",
                "make": "Ashok Leyland",
                "model": "Dost+",
                "rto_city": "Chennai South, Tamil Nadu",
                "synthetic_owner_id": "SYN-OWN-1018",
            },
            {
                "label": "Decoy 5B: Character confusion (0 vs O) + Colour shift",
                "plate_number": "TN09BY71O4",
                "body_type": "truck",
                "colour": "white",
                "make": "Eicher",
                "model": "Pro 2049",
                "rto_city": "Chennai South, Tamil Nadu",
                "synthetic_owner_id": "SYN-OWN-1019",
            },
            {
                "label": "Decoy 5C: Character confusion (1 vs I) + Body shift to Bus",
                "plate_number": "TN09BY7I04",
                "body_type": "bus",
                "colour": "blue",
                "make": "Tata",
                "model": "Starbus",
                "rto_city": "Chennai South, Tamil Nadu",
                "synthetic_owner_id": "SYN-OWN-1020",
            },
        ],
    },
]


# ===========================================================================
# SYNTHETIC DATA GENERATION PARAMETERS
# ===========================================================================

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
            ("Honda", ["City", "Amaze"]),
            ("Tata", ["Tiago", "Altroz", "Tigor"]),
            ("Volkswagen", ["Virtus", "Polo"]),
            ("Skoda", ["Slavia"]),
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
# DATABASE INITIALIZATION AND SEEDING
# ===========================================================================

def init_db(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database with optimized schema and indices."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()  # Clean re-seed

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles(plate_number);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicles_body_colour ON vehicles(body_type, colour);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicles_make_model ON vehicles(make, model);")

    conn.commit()
    return conn


def generate_synthetic_plate(existing_plates: set) -> tuple[str, str]:
    """Generate a realistic, unique Indian vehicle registration plate."""
    state, rto_num, rto_city = random.choice(RTO_REGIONS)
    
    # 1 or 2 series letters (e.g. AB or C)
    series_len = random.choice([1, 2])
    series = "".join(random.choice(SERIES_LETTERS) for _ in range(series_len))

    # 4-digit number (0001 to 9999)
    num = f"{random.randint(1, 9999):04d}"

    plate = f"{state}{rto_num}{series}{num}"
    while plate in existing_plates:
        series = "".join(random.choice(SERIES_LETTERS) for _ in range(series_len))
        num = f"{random.randint(1, 9999):04d}"
        plate = f"{state}{rto_num}{series}{num}"

    existing_plates.add(plate)
    return plate, rto_city


def generate_synthetic_owner_id(idx: int) -> str:
    """Generate 100% synthetic privacy-preserving owner token (never real personal data)."""
    random_hex = f"{random.randint(0x1000, 0xFFFF):04X}"
    return f"SYN-IND-{idx:05d}-{random_hex}"


def seed_database(db_path: Path, synthetic_target_count: int = 1000):
    """Seed the database with ground-truth test cases and realistic synthetic vehicles."""
    random.seed(42)  # Deterministic repeatability

    conn = init_db(db_path)
    cursor = conn.cursor()

    existing_plates = set()
    records_to_insert = []

    # 1. Insert Ground-Truth Vehicles & Near-Miss Decoys
    print("=" * 70)
    print("STEP 1: SEEDING GROUND-TRUTH VEHICLES & HARD DECOYS")
    print("=" * 70)

    gt_count = 0
    decoy_count = 0

    for test_block in GROUND_TRUTH_TEST_VEHICLES:
        test_case = test_block["test_case"]
        gt = test_block["ground_truth"]
        decoys = test_block["decoys"]

        print(f"\n[+] {test_case}")
        print(f"    * Ground Truth: {gt['plate_number']} | {gt['colour']} {gt['body_type']} ({gt['make']} {gt['model']}) | {gt['synthetic_owner_id']}")
        
        existing_plates.add(gt["plate_number"])
        records_to_insert.append((
            gt["plate_number"],
            gt["body_type"],
            gt["colour"],
            gt["make"],
            gt["model"],
            gt["rto_city"],
            gt["synthetic_owner_id"],
        ))
        gt_count += 1

        for d in decoys:
            print(f"      - {d['label']}: {d['plate_number']} | {d['colour']} {d['body_type']} ({d['make']} {d['model']})")
            existing_plates.add(d["plate_number"])
            records_to_insert.append((
                d["plate_number"],
                d["body_type"],
                d["colour"],
                d["make"],
                d["model"],
                d["rto_city"],
                d["synthetic_owner_id"],
            ))
            decoy_count += 1

    # 2. Generate 1,000 Realistic Synthetic Vehicles
    print("\n" + "=" * 70)
    print(f"STEP 2: GENERATING {synthetic_target_count} REALISTIC SYNTHETIC INDIAN VEHICLES")
    print("=" * 70)

    body_types = list(VEHICLE_CATALOG.keys())
    body_type_weights = [VEHICLE_CATALOG[bt]["weight"] for bt in body_types]

    colours = [c[0] for c in COLOUR_PALETTE]
    colour_weights = [c[1] for c in COLOUR_PALETTE]

    for i in range(1, synthetic_target_count + 1):
        chosen_body = random.choices(body_types, weights=body_type_weights, k=1)[0]
        makes_list = VEHICLE_CATALOG[chosen_body]["makes"]
        chosen_make, models_list = random.choice(makes_list)
        chosen_model = random.choice(models_list)
        chosen_colour = random.choices(colours, weights=colour_weights, k=1)[0]

        plate, rto_city = generate_synthetic_plate(existing_plates)
        owner_id = generate_synthetic_owner_id(i)

        records_to_insert.append((
            plate,
            chosen_body,
            chosen_colour,
            chosen_make,
            chosen_model,
            rto_city,
            owner_id,
        ))

    # Bulk insert
    cursor.executemany("""
        INSERT INTO vehicles (plate_number, body_type, colour, make, model, rto_city, synthetic_owner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """, records_to_insert)

    conn.commit()

    # Verify counts
    cursor.execute("SELECT COUNT(*) FROM vehicles;")
    total_in_db = cursor.fetchone()[0]

    cursor.execute("SELECT body_type, COUNT(*) FROM vehicles GROUP BY body_type ORDER BY COUNT(*) DESC;")
    distribution = cursor.fetchall()

    conn.close()

    print(f"\n[SUCCESS] Seeded {total_in_db} total vehicles into '{db_path}'.")
    print(f"  - Ground-truth target vehicles: {gt_count}")
    print(f"  - Hard near-miss decoys:       {decoy_count}")
    print(f"  - Background synthetic fleet:  {synthetic_target_count}")
    print("\nVehicle Fleet Distribution by Body Type:")
    for b_type, count in distribution:
        print(f"  - {b_type:<15}: {count:4d} ({count / total_in_db * 100:.1f}%)")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    db_file = base_dir / "data" / "registry.db"
    seed_database(db_file, synthetic_target_count=1000)

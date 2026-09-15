import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

EXCLUDED_LOCATION_IDS = {
    900048093, 900057108, 900057109, 900040488, 900054907, 900056339, 900058301,
    -1205055, 900040705, 900057835, 900039110, 900040163, 900040440, 900040779,
    900041008, 900049207, 900049418, 900050178, 900050887, 900050922, 900051425,
    900052343, 900052393, 900052448, 900053573, 900055417, 900055467, 900055530,
    900055688, 900057093, 900057995, 900058116, 900058370, 900059728, 900059963,
    900060594, 900061398, -670340, -200109, 664266801, -1122910, 900057257,
    900055737, 900039622, -478790, 200109,
}

def belongs_to_spain(location_id: str) -> bool:
    try:
        value = int(location_id)
    except (TypeError, ValueError):
        return False
    if -2200000 < value < -2100000: return False
    if -1500000 < value < -1390000: return False
    if value in EXCLUDED_LOCATION_IDS: return False
    return True

def fetch_markers(url: str, payload: dict, timeout: int = 20) -> Dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/json",
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    
    if response.status_code != 200:
        print(f"\n[HTTP ERROR {response.status_code}] Booking ha rechazado la petición.")
        print(f"Detalle del servidor: {response.text[:500]}")
        response.raise_for_status()
        
    data = response.json()
    
    if "errors" in data:
        print(f"\n[GraphQL ERROR] {data['errors'][0].get('message')}")
        
    results = data.get("data", {}).get("searchQueries", {}).get("search", {}).get("results", [])
    if not results:
        time.sleep(0.8)
        
    return data

def fetch_markers_with_retry(
    url: str,
    payload: dict,
    timeout: int = 20,
    retries: int = 3,
    backoff_seconds: float = 1.5
) -> Tuple[Dict, int]:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            data = fetch_markers(url, payload, timeout=timeout)
            return data, attempt
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise last_exc

def load_cells(csv_path: Path) -> List[Tuple[float, float, float, bool]]:
    encodings = ("utf-8-sig", "cp1252", "latin-1")
    last_exc = None
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.reader(f, delimiter=";")
                cells: List[Tuple[float, float, float, bool]] = []
                for row in reader:
                    if len(row) < 3: continue
                    x = float(row[0])
                    y = float(row[1])
                    inc = float(row[2])
                    mixed = len(row) > 4 and row[4].strip().lower() == "mixto"
                    cells.append((x, y, inc, mixed))
                return cells
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
    if last_exc is not None: raise last_exc
    return []

def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def print_progress(current: int, total: int, started_at: float) -> None:
    if total <= 0: total = 1
    elapsed = time.time() - started_at
    percent = current / total
    filled = int(30 * percent)
    bar = "#" * filled + "-" * (30 - filled)
    remaining = max(0.0, (elapsed / current * total) - elapsed) if current > 0 else 0.0

    message = f"\r[{bar}] {percent * 100:6.2f}% ({current}/{total}) Transcurrido: {format_duration(elapsed)} Restante: {format_duration(remaining)}"
    print(message, end="", flush=True)
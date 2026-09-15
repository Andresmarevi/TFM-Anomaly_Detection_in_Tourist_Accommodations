import argparse
import copy
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional
from alojamiento import Alojamiento
from utils import belongs_to_spain, fetch_markers_with_retry, load_cells

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://www.booking.com/dml/graphql"

def next_saturday(d: date) -> date:
    days_until_saturday = (5 - d.weekday()) % 7
    if days_until_saturday == 0: days_until_saturday = 7
    return d + timedelta(days=days_until_saturday)

def resolve_dates(period: str, today: Optional[date] = None):
    base_date = today or date.today()
    if period == "1m": target_date = base_date + timedelta(days=30)
    elif period == "3m": target_date = base_date + timedelta(days=90)
    else: raise ValueError("period must be '1m' or '3m'")

    checkin_date = next_saturday(target_date)
    checkout_date = checkin_date + timedelta(days=7)

    return base_date.isoformat(), checkin_date.isoformat(), checkout_date.isoformat()

def parse_float(node: Optional[object], default: float = 0.0) -> float:
    if node is None: return default
    try: return float(node)
    except (TypeError, ValueError): return default

def parse_int(node: Optional[object], default: int = 0) -> int:
    if node is None: return default
    try: return int(node)
    except (TypeError, ValueError): return default

def to_str(value): return None if value is None else str(value)

def to_serializable(value): 
    if value is None: return "null"
    if isinstance(value, bool): return "true" if value else "false"
    return str(value)

@dataclass
class CellResult:
    index: int
    x: float
    y: float
    inc: float
    mixed: bool
    status: str
    rows: List[Alojamiento]
    alojamientos: int
    attempts: int
    elapsed_seconds: float
    error: str = ""

def to_row(alojamiento: Alojamiento, extraction_date: str, checkin: str, checkout: str) -> List[str]:
    return [
        to_serializable(alojamiento.hotel_id), to_serializable(alojamiento.hotel_name),
        to_serializable(alojamiento.room_id), to_serializable(alojamiento.room_name),
        to_serializable(alojamiento.location_id), to_serializable(alojamiento.accommodation_type),
        to_serializable(alojamiento.num_bedrooms), to_serializable(alojamiento.num_bathrooms),
        to_serializable(alojamiento.area), to_serializable(alojamiento.stars),
        to_serializable(alojamiento.price), to_serializable(alojamiento.offer),
        to_serializable(alojamiento.price_without_offer), to_serializable(alojamiento.capacity),
        to_serializable(alojamiento.meal_plan), to_serializable(alojamiento.booking_conditions),
        to_serializable(extraction_date), to_serializable(checkin), to_serializable(checkout),
        to_serializable(alojamiento.longitude), to_serializable(alojamiento.latitude),
        to_serializable(alojamiento.piscina), to_serializable(alojamiento.primera_linea),
    ]

def build_output_line(alojamiento: Alojamiento, extraction_date: str, checkin: str, checkout: str) -> str:
    return "|;".join(to_row(alojamiento, extraction_date, checkin, checkout)) + "\n"

def safe_get_dict(value: Optional[object]) -> Dict: return value if isinstance(value, dict) else {}
def safe_get_list(value: Optional[object]) -> List: return value if isinstance(value, list) else []

def iter_alojamientos(page: Dict, mixed_cell: bool) -> Iterable[Alojamiento]:
    results = page.get("data", {}).get("searchQueries", {}).get("search", {}).get("results", [])
    
    for hotel in results:
        if not isinstance(hotel, dict): continue
        basic_data = safe_get_dict(hotel.get("basicPropertyData"))
        location = safe_get_dict(hotel.get("location"))
        
        location_id = str(basic_data.get("ufi", location.get("ufi", "")))
        if mixed_cell and (not belongs_to_spain(location_id)): continue

        alojamiento = Alojamiento()
        alojamiento.hotel_id = parse_int(basic_data.get("id"))
        alojamiento.hotel_name = to_str(safe_get_dict(hotel.get("displayName")).get("text"))
        alojamiento.location_id = location_id
        alojamiento.accommodation_type = to_str(basic_data.get("accommodationTypeId"))
        
        # Accedemos a beachDistance directamente del nodo location del hotel
        beach_dist = str(location.get("beachDistance", "")).lower()
        palabras_clave = ["frente a la playa"]
        alojamiento.primera_linea = 1 if any(p in beach_dist for p in palabras_clave) else 0

        match_config = safe_get_dict(hotel.get("matchingUnitConfigurations"))
        common_config = safe_get_dict(match_config.get("commonConfiguration"))
        alojamiento.num_bedrooms = parse_int(common_config.get("nbBedrooms"))
        alojamiento.num_bathrooms = parse_int(common_config.get("nbBathrooms"))
        
        localized_area = safe_get_dict(common_config.get("localizedArea"))
        if localized_area: alojamiento.area = to_str(localized_area.get("localizedArea"))
            
        unit_configs = safe_get_list(match_config.get("unitConfigurations"))
        if unit_configs: alojamiento.room_name = to_str(safe_get_dict(unit_configs[0]).get("name"))

        blocks = safe_get_list(hotel.get("blocks"))
        if blocks:
            first_block = safe_get_dict(blocks[0])
            block_id = safe_get_dict(first_block.get("blockId"))
            alojamiento.room_id = parse_int(block_id.get("roomId"))
            alojamiento.capacity = parse_int(block_id.get("occupancy"))
            alojamiento.meal_plan = 0
            meal_plan = first_block.get("mealPlanIncluded")
            if meal_plan is not None: alojamiento.meal_plan = 1
            policies = safe_get_dict(hotel.get("policies"))
            alojamiento.booking_conditions = 0
            if (policies.get("showFreeCancellation") is True):
                alojamiento.booking_conditions = 1

        price_info = safe_get_dict(hotel.get("priceDisplayInfoIrene"))
        amount_per_stay = safe_get_dict(safe_get_dict(price_info.get("displayPrice")).get("amountPerStay"))
        alojamiento.price = parse_float(amount_per_stay.get("amountUnformatted"))
        
        amount_before = safe_get_dict(safe_get_dict(price_info.get("priceBeforeDiscount")).get("amountPerStay"))
        if amount_before.get("amountUnformatted"):
            alojamiento.price_without_offer = parse_float(amount_before.get("amountUnformatted"))
            if alojamiento.price_without_offer > alojamiento.price:
                alojamiento.offer = alojamiento.price_without_offer - alojamiento.price

        display_location = safe_get_dict(location.get("displayLocation"))
        lat = display_location.get("latitude", location.get("latitude", basic_data.get("location", {}).get("latitude")))
        lon = display_location.get("longitude", location.get("longitude", basic_data.get("location", {}).get("longitude")))
        
        alojamiento.latitude = to_str(lat)
        alojamiento.longitude = to_str(lon)
        
        stars = safe_get_dict(basic_data.get("starRating"))
        if stars.get("value"): alojamiento.stars = to_str(stars.get("value"))

        alojamiento.piscina = 0
        yield alojamiento

def build_cell_result(
    index: int,
    x: float,
    y: float,
    inc: float,
    mixed: bool,
    checkin: str,
    checkout: str,
    extraction_date: str,
    timeout: int,
    retries: int,
    base_payload: dict,
) -> CellResult:
    started_at = time.time()
    
    payload = copy.deepcopy(base_payload)
    
    try:
        variables = payload.get("variables", {})
        inp = variables.get("input", {})
        
        if "dates" in inp:
            inp["dates"]["checkin"] = checkin
            inp["dates"]["checkout"] = checkout
            
        if "flexibleDatesConfig" in inp and "dateRangeCalendar" in inp["flexibleDatesConfig"]:
            inp["flexibleDatesConfig"]["dateRangeCalendar"]["checkin"] = [checkin]
            inp["flexibleDatesConfig"]["dateRangeCalendar"]["checkout"] = [checkout]

        if "location" in inp:
            loc = inp["location"]
            loc["hotelIds"] = [] 
            
            if "boundingBox" in loc:
                loc["boundingBox"]["swLon"] = x
                loc["boundingBox"]["swLat"] = y
                loc["boundingBox"]["neLon"] = x + inc
                loc["boundingBox"]["neLat"] = y + inc
                loc["boundingBox"]["precision"] = 1

        markers_input = variables.get("markersInput", {})
        if "boundingBox" in markers_input:
            markers_input["boundingBox"]["southWest"] = {"latitude": y, "longitude": x}
            markers_input["boundingBox"]["northEast"] = {"latitude": y + inc, "longitude": x + inc}
            markers_input["boundingBox"]["precision"] = 1
            
    except Exception as e:
        print(f"\n[ERROR] Modificando el payload: {e}")

    try:
        page, attempts = fetch_markers_with_retry(BASE_URL, payload=payload, timeout=timeout, retries=retries)
        rows = list(iter_alojamientos(page, mixed_cell=mixed))
        return CellResult(
            index=index, x=x, y=y, inc=inc, mixed=mixed, status="ok", 
            rows=rows, alojamientos=len(rows), attempts=attempts, elapsed_seconds=time.time() - started_at
        )
    except Exception as exc:
        return CellResult(
            index=index, x=x, y=y, inc=inc, mixed=mixed, status="error", 
            rows=[], alojamientos=0, attempts=retries, elapsed_seconds=time.time() - started_at, error=str(exc)
        )
        
def count_cells(csv_path: Path) -> int:
    encodings = ("utf-8-sig", "cp1252", "latin-1")
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                return sum(1 for row in csv.reader(f, delimiter=";") if len(row) >= 3)
        except UnicodeDecodeError: continue
    return 0

def build_trace_headers() -> List[str]: return ["index", "x", "y", "inc", "mixed", "status", "alojamientos", "attempts", "elapsed_seconds", "error"]
def build_trace_row(result: CellResult) -> List[str]: return [str(result.index), str(result.x), str(result.y), str(result.inc), str(result.mixed), result.status, str(result.alojamientos), str(result.attempts), f"{result.elapsed_seconds:.3f}", result.error]

def resolve_csv_path(csv_path: Path) -> Path:
    if csv_path.exists(): return csv_path
    for name in ["TodoEspañaSinCeros.csv", "TodoEspañaSinCerosPython.csv"]:
        if csv_path.with_name(name).exists(): return csv_path.with_name(name)
    raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")

def load_base_payload() -> dict:
    p = Path("src/payload.json")
    if not p.exists(): raise FileNotFoundError("Falta payload.json en el directorio.")
    with p.open("r", encoding="utf-8") as f: return json.load(f)

def run_scraping(period: str, csv_path: Path, output_dir: Path, timeout: int = 20, workers: int = 5, retries: int = 3) -> Path:
    base_payload = load_base_payload()
    extraction_date, checkin, checkout = resolve_dates(period)
    output_path = output_dir / f"Booking{extraction_date}_{period}_raw.csv"
    trace_path = output_dir.parent / "trace" / f"Booking{extraction_date}_{period}_trace.csv"
    
    csv_path = resolve_csv_path(csv_path)
    total_cells = count_cells(csv_path)
    cells = load_cells(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Procesando {total_cells} celdas...")
    results_by_index: Dict[int, CellResult] = {}
    started_at = time.time()
    processed_cells = 0

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(cells)))) as executor:
        futures = [
            executor.submit(build_cell_result, i, x, y, inc, mixed, checkin, checkout, extraction_date, timeout, retries, base_payload)
            for i, (x, y, inc, mixed) in enumerate(cells)
        ]

        for future in as_completed(futures):
            result = future.result()
            results_by_index[result.index] = result
            processed_cells += 1
            
            if result.alojamientos > 0:
                print(f"\n[Celda {processed_cells}/{total_cells}] -> OK: {result.alojamientos} hoteles encontrados.")
                primer = result.rows[0]
                print(f" ↳ Ej: 🏨 {primer.hotel_name} | {primer.price}€")
            else:
                print(f"[Celda {processed_cells}/{total_cells}] -> Buscando...".ljust(50), end="\r")

    with output_path.open("w", encoding="utf-8-sig", newline="") as out_file, trace_path.open("w", encoding="utf-8-sig", newline="") as trace_file:
        trace_writer = csv.writer(trace_file, delimiter=";")
        trace_writer.writerow(build_trace_headers())
        tot_aloj, tot_err = 0, 0

        for index in range(len(cells)):
            res = results_by_index.get(index)
            if not res: continue
            trace_writer.writerow(build_trace_row(res))
            if res.status == "error":
                tot_err += 1
                continue
            tot_aloj += res.alojamientos
            for al in res.rows: out_file.write(build_output_line(al, extraction_date, checkin, checkout))

    print(f"\nResumen: {len(cells)} celdas, {tot_aloj} alojamientos, {tot_err} errores.")    
    return output_path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("period", choices=["1m", "3m"])
    p.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "TodoEspañaSinCeros.csv"))
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "raw"))
    p.add_argument("--timeout", type=int, default=25)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Salida: {run_scraping(args.period, Path(args.csv), Path(args.output_dir), args.timeout, args.workers, args.retries)}")
# pipeline/run_pipeline.py
import subprocess
import sys
import os
import re
import shutil
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
SRC_DIR = BASE_DIR / "src"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
REPORTS_DIR = ROOT_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SCRAPER_FINAL_DIR = ROOT_DIR / "scraper" / "output" / "final"
PROCESSED_HISTORY_DIR = ROOT_DIR / "data" / "processed" / "history"
RESULTS_HISTORY_DIR = ROOT_DIR / "data" / "results" / "history"

# Scripts modulares
PIPELINE_STEPS = [
    ("etl.py", "1/3 ETL y Feature Engineering"),
    ("model.py", "2/3 Entrenamiento de Modelos"),
    ("evaluate.py", "3/3 Análisis de Validación y Consenso"),
]

# Notebooks para exportar HTML
NOTEBOOKS_TO_EXPORT = [
    "EDA_anomaly_detection.ipynb",
    "Model_Training.ipynb",
    "Anomaly_Analysis_Validation.ipynb",
]

def run_script(script_name: str, description: str, environment=None):
    script_path = SRC_DIR / script_name
    print(f"\n==================================================")
    print(f"EJECUTANDO: {description} ({script_name})")
    print(f"==================================================")
    script_environment = os.environ.copy()
    if environment:
        script_environment.update(environment)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SRC_DIR),
        text=True,
        env=script_environment,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Error crítico en {script_name}")


def archive_snapshot(input_file: Path):
    suffix = snapshot_suffix(input_file)
    PROCESSED_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    files_to_archive = [
        (ROOT_DIR / "data" / "processed" / "dataset_raw_features.csv",
         PROCESSED_HISTORY_DIR / f"dataset_raw_features_{suffix}.csv"),
        (ROOT_DIR / "data" / "processed" / "dataset_anomaly_detection_ready.csv",
         PROCESSED_HISTORY_DIR / f"dataset_anomaly_detection_ready_{suffix}.csv"),
        (ROOT_DIR / "data" / "results" / "model_results_full.csv",
         RESULTS_HISTORY_DIR / f"model_results_full_{suffix}.csv"),
    ]

    for source, destination in files_to_archive:
        if not source.exists():
            raise FileNotFoundError(f"No se pudo archivar la salida esperada: {source}")
        shutil.copy2(source, destination)
    print(f"Snapshot archivado: {suffix}")


def snapshot_suffix(input_file: Path) -> str:
    match = re.search(
        r"Booking(?P<extraction_date>\d{4}-\d{2}-\d{2})_(?P<lookahead_period>1m|3m)",
        input_file.stem,
    )
    if not match:
        raise ValueError(f"Nombre de snapshot no válido: {input_file.name}")
    return f"{match.group('extraction_date')}_{match.group('lookahead_period')}"


def select_weekly_snapshots(input_files):
    """Keep the latest extraction for each ISO week and booking horizon."""
    selected = {}
    for input_file in input_files:
        suffix = snapshot_suffix(input_file)
        extraction_date_text, lookahead_period = suffix.rsplit("_", 1)
        extraction_date = date.fromisoformat(extraction_date_text)
        week_key = (*extraction_date.isocalendar()[:2], lookahead_period)
        current = selected.get(week_key)
        if current is None or extraction_date > current[0]:
            selected[week_key] = (extraction_date, input_file)
    return [item[1] for item in sorted(selected.values(), key=lambda item: item[0])]


def snapshot_is_complete(input_file: Path) -> bool:
    suffix = snapshot_suffix(input_file)
    expected_files = [
        PROCESSED_HISTORY_DIR / f"dataset_raw_features_{suffix}.csv",
        PROCESSED_HISTORY_DIR / f"dataset_anomaly_detection_ready_{suffix}.csv",
        RESULTS_HISTORY_DIR / f"model_results_full_{suffix}.csv",
    ]
    return all(file.exists() for file in expected_files)


def process_snapshots():
    all_input_files = sorted(SCRAPER_FINAL_DIR.glob("Booking*.csv"))
    if not all_input_files:
        raise FileNotFoundError(f"No hay snapshots en: {SCRAPER_FINAL_DIR}")

    input_files = select_weekly_snapshots(all_input_files)
    weekly_ignored_count = len(all_input_files) - len(input_files)

    pending_files = [
        input_file for input_file in input_files
        if not snapshot_is_complete(input_file)
    ]
    skipped_count = len(input_files) - len(pending_files)
    print(
        f"Snapshots encontrados: {len(all_input_files)} | "
        f"Ignorados por duplicidad semanal: {weekly_ignored_count} | "
        f"Pendientes: {len(pending_files)} | Omitidos: {skipped_count}"
    )

    for index, input_file in enumerate(pending_files, start=1):
        suffix = input_file.stem.replace("Booking", "")
        environment = {"PIPELINE_INPUT_FILE": str(input_file)}
        run_script("etl.py", f"{index}/{len(pending_files)} ETL ({suffix})", environment)
        run_script("model.py", f"{index}/{len(pending_files)} Modelos ({suffix})", environment)
        archive_snapshot(input_file)

def export_notebook_reports():
    print(f"\n==================================================")
    print(f"GENERANDO REPORTES HTML EN reports/")
    print(f"==================================================")
    
    for nb_name in NOTEBOOKS_TO_EXPORT:
        nb_path = NOTEBOOKS_DIR / nb_name
        if not nb_path.exists():
            print(f"Notebook no encontrado: {nb_name} (omitido)")
            continue
            
        output_html_name = nb_path.stem + "_report"
        
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "html",
            f"--output-dir={str(REPORTS_DIR)}",
            f"--output={output_html_name}",
            str(nb_path)
        ]
        
        result = subprocess.run(cmd, cwd=str(ROOT_DIR), text=True)
        if result.returncode != 0:
            print(f"Error al exportar {nb_name}")
        else:
            print(f" Reporte generado: reports/{output_html_name}.html")

def main():
    print("Iniciando Pipeline Modular de Machine Learning...")
    process_snapshots()
    run_script("temporal_summary.py", "Resumen de evolución temporal")
    run_script("evaluate.py", "3/3 Análisis de Validación y Consenso")
        
    if os.environ.get("EXPORT_REPORTS", "1") == "1":
        export_notebook_reports()
    else:
        print("Exportación de informes omitida (EXPORT_REPORTS=0).")
    
    print("\n==================================================")
    print("PIPELINE Y REPORTES FINALIZADOS CON ÉXITO")
    print("==================================================")

if __name__ == "__main__":
    main()
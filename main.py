# main.py
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "pipeline_execution.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)

def run_phase(command: list, phase_name: str, working_dir: Path):
    """Ejecuta una fase completa del sistema con control de tiempos y errores."""
    logging.info(f"==================================================")
    logging.info(f"INICIANDO FASE: {phase_name}")
    logging.info(f"Directorio de trabajo: {working_dir}")
    logging.info(f"==================================================")
    start_time = datetime.now()

    result = subprocess.run(
        [sys.executable] + command,
        cwd=str(working_dir),
        text=True
    )

    if result.returncode != 0:
        logging.error(f"Error en la fase: {phase_name}")
        raise RuntimeError(f"Fallo en la fase: {phase_name}")

    duration = (datetime.now() - start_time).total_seconds()
    logging.info(f"COMPLETADO CON ÉXITO: {phase_name} ({duration:.2f} segundos)\n")

def execute_full_flow():
    # 1. Fase de Extracción / Scraper
    scraper_dir = BASE_DIR / "scraper"
    run_phase(["main_pipeline.py"], "1/2 Extracción y Scraping de Booking", working_dir=scraper_dir)

    # 2. Fase de Machine Learning
    pipeline_dir = BASE_DIR / "pipeline"
    run_phase(["run_pipeline.py"], "2/2 Procesamiento, Modelado y Validación ML", working_dir=pipeline_dir)

    logging.info("==================================================")
    logging.info("SISTEMA AUTOMATIZADO COMPLETADO CON ÉXITO")
    logging.info("Datos y resultados actualizados listos para el Dashboard.")
    logging.info("==================================================")

if __name__ == "__main__":
    try:
        execute_full_flow()
    except Exception as e:
        logging.critical(f"El flujo global se detuvo: {e}")
        sys.exit(1)
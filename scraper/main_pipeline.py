import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_script(script_name: str, args: list = None) -> bool:
    """Ejecuta un script de Python y devuelve True si tiene éxito."""
    command = [sys.executable, script_name]
    if args:
        command.extend(args)
        
    try:
        print(f"\nEjecutando: {' '.join(command)}")
        print("-" * 50)
        
        subprocess.run(command, check=True)
        
        print("-" * 50)
        print(f"{script_name} completado con éxito.")
        return True
        
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"ERROR: {script_name} falló con código de salida {e.returncode}.")
        return False


def main():
    print("=" * 50)
    print("   PIPELINE BOOKING - EXTRACCIÓN Y PROCESADO")
    print("=" * 50)

    start_time = time.time()

    # Leer parámetro meses
    if len(sys.argv) > 1:
        periodo = sys.argv[1]
        if periodo not in ["1m", "3m", "all"]:
            print("Error: usa '1m', '3m' o 'all'")
            sys.exit(1)
    else:
        periodo = "all"

    script_scraping = "src/booking_scraper.py"
    script_enrichment = "src/enrichment.py"

    periodos = ["1m", "3m"] if periodo == "all" else [periodo]
    today_str = datetime.now().strftime("%Y-%m-%d")

    outputs = []

    # Fase 1: Scraping Principal
    for p in periodos:
        print(f"\n--- FASE 1: Scraping Booking ({p}) ---")
        if not run_script(script_scraping, [p]):
            print(f"\nError. El pipeline se ha detenido en la Fase 1 ({p}).")
            sys.exit(1)
        csv_path = f"Booking{today_str}_{p}_raw.csv"
        outputs.append(csv_path)

    # Fase 2: Enrichment
    print("\n--- FASE 2: Enrichment de datos ---")
    
    if not run_script(script_enrichment, outputs):
        print("\nError. El pipeline se ha detenido en la Fase 2.")
        sys.exit(1)

    # Resumen
    elapsed = time.time() - start_time
    minutos, segundos = divmod(int(elapsed), 60)

    print("\n" + "=" * 50)
    print(f"PIPELINE FINALIZADO CORRECTAMENTE en {minutos}m {segundos}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
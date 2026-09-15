import pandas as pd
import csv
from pathlib import Path
import sys
import json

# Limpiar dobles saltos de línea (CSV principal)
def clean_double_newlines(input_path: Path, output_path: Path):
    """
    Elimina líneas vacías (dobles saltos de línea)
    """
    with open(input_path, "r", encoding="utf-8-sig", errors="replace") as f_in:
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f_out:
            for line in f_in:
                # Si la línea no es solo un salto de línea vacío, la escribimos
                if line not in ['\n', '\r\n']:
                    f_out.write(line)


# Proceso principal
def process_booking(
    input_csv: Path,
    piscina_csv: Path,
    localizaciones_csv: Path,
    output_csv: Path,
):

    # Leer CSV forzando formato texto y vacíos literales
    df = pd.read_csv(input_csv, sep="\\|;", engine="python", dtype=str, keep_default_na=False, encoding="utf-8-sig")

    df.columns = [
        "id",
        "nombre",
        "idHabitacion",
        "nombreHabitacion",
        "idLocalizacion",
        "tipo",
        "numHabitaciones",
        "numBanos",
        "metros",
        "estrellas",
        "precio",
        "oferta",
        "precioSinOferta",
        "capacidad",
        "desayuno",
        "cancelacion",
        "fechaActual",
        "fechaEntrada",
        "fechaSalida",
        "coordenadaX",
        "coordenadaY",
        "piscina",
        "primeraLinea",
    ]
    
    # Eliminar filas corruptas sin id válido
    df["id"] = df["id"].astype(str).str.strip()

    # Eliminar filas donde id está vacío
    df = df[df["id"] != ""]

    # Eliminar filas donde id no es numérico (ej: -372490, textos, etc.)
    df = df[df["id"].str.match(r"^\d+$", na=False)]

    # Leer CSV de piscinas
    piscina = pd.read_csv(
        piscina_csv,
        sep=";",
        encoding="latin",
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip"
    )
    piscina.columns = ["id", "nombre", "idLocal", "tipo"]
    
    # Leer CSV de localizaciones
    loc = pd.read_csv(
        localizaciones_csv,
        sep=";",
        encoding="latin-1",
        header=None,
        dtype=str,
        keep_default_na=False,
        names=[
            "idLocalizacion",
            "nombreCCAA",
            "nombreProvincia",
            "nombreMunicipio",
            "nombreLocalizacion",
        ]
    )

    # Eliminar duplicados
    piscina = piscina.drop_duplicates(subset=["id"])
    loc = loc.drop_duplicates(subset=["idLocalizacion"])

    # Merge localizaciones
    df = df.merge(loc, how="left", on="idLocalizacion")

    # Limpieza de IDs para el cruce
    df["id"] = df["id"].str.replace(".0", "", regex=False).str.strip()
    piscina["id"] = piscina["id"].str.strip()

    df["id"] = df["id"].str.replace(".0", "", regex=False).str.strip()
    piscina["id"] = piscina["id"].str.strip()
    
    df["piscina_flag"] = df["id"].isin(piscina["id"]).astype(int).astype(str)
    df["playa_flag"] = (
        df["primeraLinea"]
        .fillna("0")
        .astype(str)
        .str.strip()
        .isin(["1", "True", "true", "TRUE"])
        .astype(int)
        .astype(str)
    )
    
    type_dict_path = PROJECT_ROOT / "data" / "accommodation_type_dict.json"
    with open(type_dict_path, "r", encoding="utf-8") as f:
        accommodation_dict = json.load(f)
    tipo_original = df["tipo"].astype(str)
    unknown_ids = sorted(
        set(tipo_original) - set(accommodation_dict.keys())
        )
    if unknown_ids:
        print("\nIDs de tipo desconocidos:")
        print(unknown_ids)
    df["tipo"] = (
        tipo_original
        .map(accommodation_dict)
        .fillna(tipo_original))

    # Limpiezas
    df["nombre"] = df["nombre"].str.replace(";", " ", regex=False)
    df.loc[df["estrellas"] == "n", "estrellas"] = ""
    
    df["idHabitacion"] = (
        df["idHabitacion"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )
    df = df[df["idHabitacion"].str.match(r"^\d+$", na=False)]
    df = df[df["idHabitacion"] != "0"]
    
    df = df[df["precio"] != "0"]
    df["capacidad"] = 2

    # Selección final
    df_final = df[
        [
            "id",
            "nombre",
            "idHabitacion",
            "nombreHabitacion",
            "idLocalizacion",
            "nombreCCAA",
            "nombreProvincia",
            "nombreMunicipio",
            "nombreLocalizacion",
            "tipo",
            "numHabitaciones",
            "numBanos",
            "metros",
            "piscina_flag",
            "playa_flag",
            "estrellas",
            "precio",
            "oferta",
            "precioSinOferta",
            "capacidad",
            "desayuno",
            "cancelacion",
            "fechaActual",
            "fechaEntrada",
            "fechaSalida",
        ]
    ].copy()

    df_final = df_final.replace("", " ")
    df_final = df_final.sort_values(by="id")

    # Exportar con formato: separador ';', con cabecera, encoding latin-1
    df_final.to_csv(
        output_csv,
        sep=";",
        index=False,
        header=True,
        encoding="latin-1",
        errors="replace"
    )

    print(f"Archivo generado: {output_csv}")
    print("Filas totales:", len(df_final))


# MAIN
if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    base = PROJECT_ROOT / "output"
    raw_dir = base / "raw"
    final_dir = base / "final"

    piscina = PROJECT_ROOT / "data" / "Piscinas.csv"
    localizaciones = PROJECT_ROOT / "data" / "localizaciones.csv"

    input_files = sys.argv[1:]

    if input_files:
        booking_files = [raw_dir / f for f in input_files]
        print("Modo pipeline: usando archivos generados en esta ejecución")
    else:
        booking_files = [
            f for f in raw_dir.glob("Booking*.csv")
            if not any(x in f.stem for x in ["_clean", "_trace"])
        ]
        print(f"Archivos encontrados: {len(booking_files)}")

    for raw in booking_files:
        print(f"\nProcesando: {raw.name}")
       
        clean = raw_dir / f"{raw.stem}_clean.csv"
        
        output_name = raw.stem.removesuffix("_raw")
        output = final_dir / f"{output_name}.csv"

        # Limpiar CSV
        clean_double_newlines(raw, clean)

        # Procesamiento
        process_booking(
            input_csv=clean,
            piscina_csv=piscina,
            localizaciones_csv=localizaciones,
            output_csv=output,
        )
        
        clean.unlink(missing_ok=True)
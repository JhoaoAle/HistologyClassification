"""
Script for downloading, extracting, organizing, and verifying a dataset
for a 3-class classification problem:
1-Tumores, 2-Estroma, 3-Inflamación benigna.
Dumps metadata into a JSON file dataset_info.json.
"""

import os
import sys
import yaml
import json
import gdown
import zipfile
import shutil
from pathlib import Path
import argparse
import logging
from datetime import datetime

# ----------------------------------
# LOGGING
# ----------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------
# CLASS MAPPING
# ----------------------------------
CLASS_MAPPING = {
    "1": "tumores",
    "2": "estroma",
    "3": "inflamacion_benigna"
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ----------------------------------
# LOAD CONFIG
# ----------------------------------
def load_config(config_path="config/config.yml"):
    try:
        with open(config_path, "r") as file:
            cfg = yaml.safe_load(file)
        logger.info(f"Configuración cargada desde: {config_path}")
        return cfg
    except FileNotFoundError:
        logger.error("No se encontró archivo de configuración.")
        sys.exit(1)


# ----------------------------------
# CREATE DIRECTORY STRUCTURE
# ----------------------------------
def setup_directories(paths):
    logger.info("Creando estructura de directorios...")

    dirs = []

    for split in ["train", "test"]:
        for _, class_name in CLASS_MAPPING.items():
            dirs.append(Path(paths["data"]["raw"]) / split / class_name)

    dirs.extend([
        Path(paths["data"]["processed"]),
        Path(paths["data"]["augmented"]),
        Path(paths["models"]["checkpoints"]),
        Path(paths["results"]["training_curves"]),
        Path(paths["results"]["gradcam"]),
        Path(paths["results"]["confusion_matrices"]),
    ])

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Estructura creada.")


# ----------------------------------
# DOWNLOAD DATA
# ----------------------------------
def download_data(downloads, temp_dir):
    logger.info("Descargando datos desde Google Drive...")

    zip_files = list(temp_dir.glob("*.zip"))
    if zip_files:
        logger.info("Archivos ZIP ya existen en temp_downloads. Saltando descarga.")
        return

    files_to_get = {}
    for name, file_id in downloads.items():
        if file_id is not None:
            files_to_get[f"{name}.zip"] = file_id

    if not files_to_get:
        logger.warning("No hay IDs de descarga en config.yml")
        return

    for filename, file_id in files_to_get.items():
        output_path = temp_dir / filename
        url = f"https://drive.google.com/uc?id={file_id}"

        logger.info(f"Descargando {filename}...")
        gdown.download(url, str(output_path), quiet=False)

        if not output_path.exists():
            logger.error(f"Fallo al descargar {filename}")
            sys.exit(1)

    logger.info("Descargas completadas.")


# ----------------------------------
# EXTRACT ZIP FILES (DETERMINISTIC)
# ----------------------------------
def extract_files(temp_dir):
    logger.info("Extrayendo archivos ZIP...")

    extract_dir = temp_dir / "extracted"

    # If extracted folder exists, skip extraction
    if extract_dir.exists():
        logger.info("Carpeta 'extracted' ya existe. Saltando extracción.")
        return

    extract_dir.mkdir(exist_ok=True)

    # Expect zips named "train.zip" and "test.zip"
    zip_paths = list(temp_dir.glob("*.zip"))
    if not zip_paths:
        logger.warning("No hay archivos ZIP para extraer.")
        return

    for zip_path in zip_paths:
        # Determine deterministic output folder
        if "train" in zip_path.stem.lower():
            outdir = extract_dir / "train"
        elif "test" in zip_path.stem.lower():
            outdir = extract_dir / "test"
        else:
            outdir = extract_dir / zip_path.stem  # fallback

        outdir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extrayendo {zip_path.name} en {outdir}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(outdir)

    logger.info("Extracción completada.")


# ----------------------------------
# FIND REAL ROOT (ROBUST)
# ----------------------------------
def find_real_root(base, depth=5):
    """
    Busca dentro de una estructura desconocida hasta encontrar
    carpetas: 1, 2, 3 (las clases).
    """
    base = Path(base)

    # Depth-first search
    def dfs(path, d):
        if d > depth:
            return None

        # Is this the real root?
        if all((path / cid).exists() for cid in CLASS_MAPPING.keys()):
            return path

        # Recurse
        for sub in path.iterdir():
            if sub.is_dir():
                result = dfs(sub, d + 1)
                if result:
                    return result
        return None

    result = dfs(base, 0)
    return result if result else base


# ----------------------------------
# ORGANIZE DATA
# ----------------------------------
def organize_data(source_dir, target_dir):
    logger.info(f"Organizando datos desde {source_dir}...")

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        logger.error(f"Directorio no encontrado: {source_dir}")
        return 0

    total = 0

    for class_id, class_name in CLASS_MAPPING.items():
        src = source_dir / class_id
        dst = target_dir / class_name

        if not src.exists():
            logger.warning(f"No existe carpeta {src}")
            continue

        images = [f for f in src.glob("*") if f.suffix.lower() in IMAGE_EXTS]

        for img in images:
            shutil.copy2(img, dst / img.name)
            total += 1

    logger.info(f"Total copiado: {total}")
    return total


# ----------------------------------
# VERIFY + SIZE STATS
# ----------------------------------
def verify_organization(root):
    logger.info("Verificando organización y calculando tamaños...")

    root = Path(root)
    stats = {
        "train": {},
        "test": {},
        "sizes": {"train": 0, "test": 0}
    }

    for split in ["train", "test"]:
        for _, class_name in CLASS_MAPPING.items():
            class_dir = root / split / class_name
            files = [f for f in class_dir.glob("*") if f.suffix.lower() in IMAGE_EXTS]

            class_info = []
            total_size = 0

            for f in files:
                size = f.stat().st_size
                class_info.append({
                    "file": f.name,
                    "size_mb": round(size / (1024 ** 2), 4)
                })
                total_size += size

            stats[split][class_name] = {
                "count": len(files),
                "size_mb": round(total_size / (1024 ** 2), 2)
            }

            stats["sizes"][split] += total_size

            logger.info(f"{split}/{class_name}: {len(files)} imágenes, {round(total_size/1048576,2)} MB")

    logger.info("Verificación completada.")
    return stats


# ----------------------------------
# SAVE JSON INFO
# ----------------------------------
def save_dataset_info(paths, stats):
    info = {
        "project_name": "Cancer Tissue Classification",
        "classes": CLASS_MAPPING,
        "stats": {
            "train": stats["train"],
            "test": stats["test"],
            "total_train_images": sum(stats["train"][c]["count"] for c in stats["train"]),
            "total_test_images": sum(stats["test"][c]["count"] for c in stats["test"]),
            "total_train_size_mb": round(stats["sizes"]["train"] / (1024 ** 2), 2),
            "total_test_size_mb": round(stats["sizes"]["test"] / (1024 ** 2), 2),
            "total_size_mb": round((stats["sizes"]["train"] + stats["sizes"]["test"]) / (1024 ** 2), 2),
        },
        "organization_date": str(datetime.now())
    }

    out_path = Path(paths["data"]["raw"]) / "dataset_info.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    logger.info(f"dataset_info.json guardado en {out_path}")


# ----------------------------------
# MAIN
# ----------------------------------
def main():
    parser = argparse.ArgumentParser(description="Setup de datos para 3 clases")
    parser.add_argument("--config", default="config/config.yml")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--data-dir")
    parser.add_argument("--temp-dir", default="temp_downloads")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]

    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(exist_ok=True)

    try:
        setup_directories(paths)

        # Download only if zip doesn't already exist
        if not args.skip_download:
            download_data(config["downloads"], temp_dir)
            extract_files(temp_dir)

            # Use deterministic extraction
            train_root = find_real_root(temp_dir / "extracted" / "train")
            test_root = find_real_root(temp_dir / "extracted" / "test")

            organize_data(train_root, Path(paths["data"]["raw"]) / "train")
            organize_data(test_root, Path(paths["data"]["raw"]) / "test")

        elif args.data_dir:
            organize_data(Path(args.data_dir) / "train", Path(paths["data"]["raw"]) / "train")
            organize_data(Path(args.data_dir) / "test", Path(paths["data"]["raw"]) / "test")

        stats = verify_organization(paths["data"]["raw"])
        save_dataset_info(paths, stats)

        if temp_dir.exists() and not args.skip_download:
            shutil.rmtree(temp_dir)

        logger.info("Setup completado")

    except Exception as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
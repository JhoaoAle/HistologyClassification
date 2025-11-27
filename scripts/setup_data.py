"""
Script para descargar y organizar datos del proyecto de detección de cáncer
Con 3 clases: 0-Tumores, 1-Estroma, 2-Inflamación benigna
Versión para ejecución local
"""

import os
import sys
import yaml
import gdown
import zipfile
import shutil
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapeo de clases según la nueva información
CLASS_MAPPING = {
    '0': 'tumores',
    '1': 'estroma', 
    '2': 'inflamacion_benigna'
}

def load_config(config_path="config/config.yaml"):
    """Carga la configuración del proyecto"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuración cargada desde: {config_path}")
        return config
    except FileNotFoundError:
        # Intentar con .yml si .yaml no existe
        try:
            with open(config_path.replace('.yaml', '.yml'), 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuración cargada desde: {config_path.replace('.yaml', '.yml')}")
            return config
        except FileNotFoundError:
            logger.error(f"No se encontró el archivo de configuración: {config_path}")
            sys.exit(1)

def setup_directories(paths_config):
    """Crea la estructura de directorios del proyecto para 3 clases"""
    logger.info("Creando estructura de directorios para 3 clases...")
    
    directories = []
    
    # Crear estructura para train y test con las 3 clases
    for split in ['train', 'test']:
        for class_id, class_name in CLASS_MAPPING.items():
            directories.append(
                Path(paths_config['data']['raw']) / split / class_name
            )
    
    # Directorios adicionales
    additional_dirs = [
        Path(paths_config['data']['processed']),
        Path(paths_config['data']['augmented']),
        Path(paths_config['models']['checkpoints']),
        Path(paths_config['results']['training_curves']),
        Path(paths_config['results']['gradcam']),
        Path(paths_config['results']['confusion_matrices']),
    ]
    
    directories.extend(additional_dirs)
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Creado: {directory}")
    
    logger.info("Estructura de directorios para 3 clases creada")

def download_data(download_config, temp_dir):
    """Descarga los archivos desde Google Drive a directorio temporal local"""
    logger.info("Descargando datos...")
    
    # Configuración de descargas
    downloads = [
        {
            'url': 'https://drive.google.com/uc?id=1c2mrPeND5-jxL43Im8eDi5PCyuTGBLEX',
            'output': temp_dir / 'train.zip',
            'description': 'Datos de entrenamiento'
        },
        {
            'url': 'https://drive.google.com/uc?id=1n_zGW0qht4rQauar3hQkqpz8sreb0SRv',
            'output': temp_dir / 'test.zip',
            'description': 'Datos de prueba'
        }
    ]
    
    for item in downloads:
        try:
            logger.info(f"Descargando: {item['description']}")
            # Usar gdown con la ruta local
            gdown.download(item['url'], str(item['output']), quiet=False)
            if item['output'].exists():
                logger.info(f"Descargado: {item['output']}")
            else:
                logger.error(f"Falló la descarga: {item['output']}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Error descargando {item['description']}: {e}")
            sys.exit(1)

def extract_files(temp_dir):
    """Extrae los archivos ZIP descargados"""
    logger.info("Extrayendo archivos...")
    
    extract_dir = temp_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    
    zip_files = [
        (temp_dir / 'train.zip', extract_dir / 'train'),
        (temp_dir / 'test.zip', extract_dir / 'test')
    ]
    
    for zip_path, extract_path in zip_files:
        if not zip_path.exists():
            logger.error(f"Archivo ZIP no encontrado: {zip_path}")
            continue
            
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            logger.info(f"Extraído: {zip_path} → {extract_path}")
        except Exception as e:
            logger.error(f"Error extrayendo {zip_path}: {e}")
            sys.exit(1)

def organize_data(source_dir, target_base_dir):
    """
    Organiza los datos en la estructura del proyecto para 3 clases
    
    Args:
        source_dir: Directorio fuente con las carpetas 0, 1, 2
        target_base_dir: Directorio base de destino
    """
    logger.info(f"Organizando datos desde {source_dir}...")
    
    source_dir = Path(source_dir)
    target_base_dir = Path(target_base_dir)
    
    if not source_dir.exists():
        logger.error(f"Directorio fuente no existe: {source_dir}")
        return 0
    
    images_copied = 0
    
    # Procesar cada una de las 3 clases
    for class_id, class_name in CLASS_MAPPING.items():
        source_class_dir = source_dir / class_id  # 0, 1, 2
        target_class_dir = target_base_dir / class_name  # tumores, estroma, inflamacion_benigna
        
        if not source_class_dir.exists():
            logger.warning(f"No se encontró directorio de clase: {source_class_dir}")
            continue
            
        logger.info(f"Procesando clase {class_id} → {class_name}")
            
        # Copiar archivos de imagen
        image_files = list(source_class_dir.glob('*.*'))
        for img_file in image_files:
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                shutil.copy2(img_file, target_class_dir / img_file.name)
                images_copied += 1
                
                if images_copied % 100 == 0:  # Log cada 100 imágenes
                    logger.info(f"Copiadas {images_copied} imágenes...")
        
        logger.info(f"  Clase {class_name}: {len(image_files)} imágenes")
    
    logger.info(f"Organización completada. Total: {images_copied} imágenes copiadas")
    return images_copied

def verify_organization(target_base_dir):
    """Verifica que los datos se hayan organizado correctamente para las 3 clases"""
    logger.info("Verificando organización de datos...")
    
    target_base_dir = Path(target_base_dir)
    stats = {}
    
    for split in ['train', 'test']:
        stats[split] = {}
        for class_name in CLASS_MAPPING.values():
            class_dir = target_base_dir / split / class_name
            image_count = len(list(class_dir.glob('*.*')))
            stats[split][class_name] = image_count
            logger.info(f"   {split}/{class_name}: {image_count} imágenes")
    
    # Resumen
    total_train = sum(stats['train'].values())
    total_test = sum(stats['test'].values())
    
    logger.info(f"Resumen final:")
    logger.info(f"   Total train: {total_train} imágenes")
    logger.info(f"   Total test: {total_test} imágenes") 
    logger.info(f"   Total general: {total_train + total_test} imágenes")
    
    # Verificar balance de clases
    if total_train > 0:
        logger.info("Balance de clases (train):")
        for class_name, count in stats['train'].items():
            percentage = (count / total_train) * 100
            logger.info(f"   {class_name}: {count} imágenes ({percentage:.1f}%)")
    
    return stats

def save_dataset_info(paths_config, stats):
    """Guarda información del dataset organizado"""
    dataset_info = {
        "project_name": "Cancer Tissue Classification",
        "classes": CLASS_MAPPING,
        "stats": {
            "train": stats['train'],
            "test": stats['test'],
            "total_train": sum(stats['train'].values()),
            "total_test": sum(stats['test'].values())
        },
        "organization_date": str(datetime.now())
    }
    
    info_path = Path(paths_config['data']['raw']) / "dataset_info.json"
    
    try:
        import json
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Información del dataset guardada en: {info_path}")
    except Exception as e:
        logger.warning(f"No se pudo guardar dataset_info: {e}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Setup de datos para proyecto de detección de cáncer (3 clases)')
    parser.add_argument('--config', default='config/config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--skip-download', action='store_true', help='Saltar descarga (solo organizar datos existentes)')
    parser.add_argument('--data-dir', help='Directorio con datos ya descargados')
    parser.add_argument('--temp-dir', default='temp_downloads', help='Directorio temporal para descargas')
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    paths = config['paths']
    
    # Crear directorio temporal local
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Crear estructura de directorios para 3 clases
        setup_directories(paths)
        
        if not args.skip_download:
            # 2. Descargar datos a directorio temporal local
            download_data(config.get('downloads', {}), temp_dir)
            
            # 3. Extraer archivos
            extract_files(temp_dir)
            
            # 4. Organizar datos de entrenamiento
            organize_data(
                source_dir=temp_dir / "extracted" / "train",
                target_base_dir=Path(paths['data']['raw']) / "train"
            )
            
            # 5. Organizar datos de prueba
            organize_data(
                source_dir=temp_dir / "extracted" / "test", 
                target_base_dir=Path(paths['data']['raw']) / "test"
            )
        elif args.data_dir:
            # Modo: solo organizar datos existentes
            logger.info(f"Organizando datos existentes desde: {args.data_dir}")
            
            # Asumiendo que args.data_dir tiene subcarpetas train/0,1,2 y test/0,1,2
            organize_data(
                source_dir=Path(args.data_dir) / "train",
                target_base_dir=Path(paths['data']['raw']) / "train"
            )
            
            organize_data(
                source_dir=Path(args.data_dir) / "test", 
                target_base_dir=Path(paths['data']['raw']) / "test"
            )
        else:
            logger.info("Saltando descarga (modo sin datos)")
        
        # 6. Verificar organización
        stats = verify_organization(paths['data']['raw'])
        
        # 7. Guardar información del dataset
        save_dataset_info(paths, stats)
        
        # 8. Limpiar temporal (opcional)
        if temp_dir.exists() and not args.skip_download:
            shutil.rmtree(temp_dir)
            logger.info(f"Directorio temporal limpiado: {temp_dir}")
        
        logger.info("Setup de datos para 3 clases completado exitosamente!")
        
    except Exception as e:
        logger.error(f"Error durante el setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
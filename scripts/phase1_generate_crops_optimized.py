#!/usr/bin/env python3
# scripts/phase1_generate_crops_optimized.py

import sys
from pathlib import Path
import json
import cv2
import yaml
from tqdm import tqdm
from datetime import datetime

# Añadir raíz al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.preprocessing.clustered_crop_generator_optimized import ClusteredCropGeneratorOptimized
from src.core.preprocessing.grid_crop_generator import GridCropGenerator
from src.core.preprocessing.vehicle_detector import VehicleDetector

def get_vehicle_zone(filename: str) -> str:
    """Extrae la zona del nombre del archivo (ej: zona1_ok...)"""
    parts = filename.split('_')
    for part in parts:
        if part.startswith('zona') and part[4:].isdigit():
            return part[4:] # Retorna "1", "2", etc.
    return "unknown"

def scan_dataset_structure(dataset_dir: Path):
    """
    Separa imágenes en damaged/clean basándose en la existencia 
    del JSON de etiquetas (_labelDANO_modificado.json).
    Lógica portada de scripts/03d_generate_hybrid_embeddings...
    """
    all_images = list(dataset_dir.glob("*_imageDANO_original.jpg"))
    
    damaged_imgs = []
    clean_imgs = []
    
    for img_path in all_images:
        # La clave para saber si tiene daño real documentado es este JSON específico
        json_damage_path = img_path.parent / img_path.name.replace(
            '_imageDANO_original.jpg',
            '_labelDANO_modificado.json'
        )
        
        if json_damage_path.exists():
            damaged_imgs.append((img_path, json_damage_path))
        else:
            clean_imgs.append(img_path)
            
    return damaged_imgs, clean_imgs

def main():
    # --- CONFIGURACIÓN ---
    dataset_dir = Path("data/raw/masked_dataset_split_2")
    output_base = Path("data/processed/crops/balanced_optimized_v2")
    
    # IMPORTANTE: Aquí definimos el tamaño base. 
    # NOTA: Debatiremos si 448 es ideal para CLIP o si debería ser 336 o 224 en el siguiente paso.
    CROP_SIZE = 336  # Resolución nativa para ViT-L/14@336px
    
    # Crear directorios
    (output_base / "damaged").mkdir(parents=True, exist_ok=True)
    (output_base / "clean").mkdir(parents=True, exist_ok=True)
    
    # --- INICIALIZACIÓN DE MÓDULOS ---
    print(f"🔧 Inicializando generadores (Crop Size: {CROP_SIZE})...")
    
    # 1. Generador para Daños (Clusterizado)
    cluster_gen = ClusteredCropGeneratorOptimized(target_size=CROP_SIZE)
    
    # 2. Generador para Limpios (Grid)
    grid_gen = GridCropGenerator(crop_size=CROP_SIZE, overlap=0.25)
    
    # 3. Detector de Vehículos (Para enfocar el grid)
    detector = VehicleDetector(model_name="yolov8n.pt", confidence=0.5)
    
    # --- ESCANEO DEL DATASET ---
    print(f"\n📂 Escaneando dataset en {dataset_dir}...")
    damaged_list, clean_list = scan_dataset_structure(dataset_dir)
    
    print(f"   - Imágenes CON daño (tienen JSON label): {len(damaged_list)}")
    print(f"   - Imágenes SIN daño (Grid + Detector): {len(clean_list)}")
    print(f"   - Total imágenes: {len(damaged_list) + len(clean_list)}\n")
    
    metadata_list = []
    
    # --- PROCESO 1: IMÁGENES CON DAÑO ---
    print(f"🚀 Procesando imágenes CON daño...")
    for img_path, json_path in tqdm(damaged_list, desc="Damaged Pipeline"):
        try:
            # Llamada al generador clusterizado (aprovecha el JSON)
            crops = cluster_gen.generate_crops(img_path, json_path)
            
            for i, crop_data in enumerate(crops):
                # Naming único
                crop_filename = f"{img_path.stem}_cluster_{i:03d}.jpg"
                save_path = output_base / "damaged" / crop_filename
                
                # Guardar imagen
                cv2.imwrite(str(save_path), crop_data['crop'])
                
                # Guardar metadata
                metadata_list.append({
                    'crop_id': crop_filename,
                    'crop_path': str(save_path),
                    'source_image': img_path.name,
                    'has_damage': True,
                    'vehicle_zone': get_vehicle_zone(img_path.name),
                    'generation_method': 'clustered_bbox',
                    'defects': crop_data.get('defects', []), # Importante para contexto posterior
                    'scale_info': crop_data.get('scale_info', {})
                })
        except Exception as e:
            print(f"❌ Error en {img_path.name}: {e}")

    # --- PROCESO 2: IMÁGENES SIN DAÑO ---
    print(f"\n🚀 Procesando imágenes SIN daño...")
    for img_path in tqdm(clean_list, desc="Clean Pipeline"):
        try:
            # 1. Detectar vehículo para no hacer crops del asfalto o cielo
            detection = detector.detect(img_path)
            
            vehicle_bbox = None
            detection_score = 0.0
            
            if detection:
                vehicle_bbox = detection['bbox']
                detection_score = detection['score']
            
            # 2. Generar Grid crops dentro del bbox detectado
            crops = grid_gen.generate_crops(img_path, vehicle_bbox=vehicle_bbox)
            
            for i, crop_data in enumerate(crops):
                # Naming único con coordenadas grid
                crop_filename = f"{img_path.stem}_grid_{crop_data['grid_x']}_{crop_data['grid_y']}.jpg"
                save_path = output_base / "clean" / crop_filename
                
                cv2.imwrite(str(save_path), crop_data['crop'])
                
                metadata_list.append({
                    'crop_id': crop_filename,
                    'crop_path': str(save_path),
                    'source_image': img_path.name,
                    'has_damage': False,
                    'vehicle_zone': get_vehicle_zone(img_path.name),
                    'generation_method': 'grid_vehicle_detected',
                    'vehicle_bbox_conf': detection_score,
                    'grid_pos': {'x': crop_data['grid_x'], 'y': crop_data['grid_y']}
                })
                
        except Exception as e:
            print(f"❌ Error en {img_path.name}: {e}")

    # --- FINALIZACIÓN ---
    meta_path = output_base.parent / "metadata_crops_phase1.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
        
    print(f"\n✅ Fase 1 Completada.")
    print(f"   - Total crops generados: {len(metadata_list)}")
    print(f"   - Metadata guardada en: {meta_path}")

if __name__ == "__main__":
    main()
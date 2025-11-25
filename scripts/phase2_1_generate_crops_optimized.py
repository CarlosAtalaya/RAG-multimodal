#!/usr/bin/env python3
# scripts/phase1_generate_crops_optimized.py

import sys
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm

# Añadir raíz al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.preprocessing.clustered_crop_generator_optimized import ClusteredCropGeneratorOptimized
from src.core.preprocessing.grid_crop_generator import GridCropGenerator

# --- FUNCIONES AUXILIARES ---

def get_vehicle_zone(filename: str) -> str:
    """Extrae la zona del nombre del archivo (ej: zona1_ok...)"""
    parts = filename.split('_')
    for part in parts:
        if part.startswith('zona') and part[4:].isdigit():
            return part[4:] 
    return "unknown"

def load_polygons_map(json_path: Path) -> dict:
    """
    Carga el fichero maestro de polígonos y crea un mapa de acceso rápido.
    Returns:
        dict: {'filename.jpg': [[x,y], [x,y], ...]}
    """
    if not json_path.exists():
        print(f"⚠️  ALERTA: No se encontró el fichero maestro: {json_path}")
        return {}
    
    print(f"📂 Cargando mapa de polígonos desde: {json_path.name}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Convertimos la lista en un diccionario para búsqueda O(1) por nombre de archivo
        # data es una lista de dicts: [{'filename': '...', 'polygon': [...]}, ...]
        poly_map = {item['filename']: item['polygon'] for item in data}
        
        print(f"   ✅ {len(poly_map)} polígonos cargados en memoria.")
        return poly_map
        
    except Exception as e:
        print(f"❌ Error crítico leyendo {json_path}: {e}")
        return {}

def scan_dataset_structure(dataset_dir: Path):
    """Separa imágenes en damaged/clean basándose en la existencia del JSON de defectos."""
    # Buscamos todas las imágenes .jpg
    all_images = list(dataset_dir.glob("*.jpg"))
    damaged_imgs = []
    clean_imgs = []
    
    for img_path in all_images:
        # JSON de defectos (específico para imágenes con daño)
        # Asumiendo nomenclatura: nombre.jpg -> nombre_labelDANO_modificado.json 
        # OJO: Ajusta este replace si tu nomenclatura varía.
        # Basado en tu ejemplo: "zona1_ko...imageDANO_original.jpg" -> comprueba si existe json de daño
        
        # Estrategia robusta: buscar si existe el fichero de labels de daño
        json_damage_name = img_path.name.replace('_imageDANO_original.jpg', '_labelDANO_modificado.json')
        if json_damage_name == img_path.name: # Si no hizo replace (naming diferente)
             json_damage_name = img_path.stem + '_labelDANO_modificado.json'
             
        json_damage_path = img_path.parent / json_damage_name
        
        if json_damage_path.exists():
            damaged_imgs.append((img_path, json_damage_path))
        else:
            clean_imgs.append(img_path)
            
    return damaged_imgs, clean_imgs

# --- MAIN ---

def main():
    # --- CONFIGURACIÓN ---
    # Actualizado a la ruta que me indicaste
    dataset_dir = Path("data/raw/masked_dataset_split_2")
    output_base = Path("data/processed/crops/balanced_optimized_v2")
    
    # Ruta al fichero maestro de polígonos
    polygons_json_path = dataset_dir / "poligonos.json"
    
    CROP_SIZE = 336  # ✅ Resolución nativa MetaCLIP/CLIP
    
    (output_base / "damaged").mkdir(parents=True, exist_ok=True)
    (output_base / "clean").mkdir(parents=True, exist_ok=True)
    
    print(f"🔧 Inicializando generadores (Crop Size: {CROP_SIZE})...")
    
    # 1. Cargar Mapa de Polígonos (NUEVO)
    polygons_map = load_polygons_map(polygons_json_path)
    
    # 2. Inicializar Generadores
    cluster_gen = ClusteredCropGeneratorOptimized(target_size=CROP_SIZE)
    grid_gen = GridCropGenerator(crop_size=CROP_SIZE, overlap=0.30, min_vehicle_ratio=0.40)
    
    print(f"\n📂 Escaneando dataset en {dataset_dir}...")
    damaged_list, clean_list = scan_dataset_structure(dataset_dir)
    
    print(f"   - Damaged (con etiquetas de defecto): {len(damaged_list)}")
    print(f"   - Clean (para grid cropping): {len(clean_list)}")
    
    metadata_list = []
    
    # --- PROCESO 1: DAMAGED (Clusterizado por Defectos) ---
    print(f"\n🚀 Procesando imágenes CON daño...")
    for img_path, json_path in tqdm(damaged_list, desc="Damaged Pipeline"):
        try:
            crops = cluster_gen.generate_crops(img_path, json_path)
            
            for i, crop_data in enumerate(crops):
                crop_filename = f"{img_path.stem}_cluster_{i:03d}.jpg"
                save_path = output_base / "damaged" / crop_filename
                
                cv2.imwrite(str(save_path), crop_data['crop'])
                
                metadata_list.append({
                    'crop_id': crop_filename,
                    'crop_path': str(save_path),
                    'source_image': img_path.name,
                    'has_damage': True,
                    'vehicle_zone': get_vehicle_zone(img_path.name),
                    'generation_method': 'clustered_defects',
                    'defects': crop_data.get('defects', []),
                    'scale_info': crop_data.get('scale_info', {})
                })
        except Exception as e:
            print(f"❌ Error en {img_path.name}: {e}")

    # --- PROCESO 2: CLEAN (Grid basado en Polígonos.json) ---
    print(f"\n🚀 Procesando imágenes SIN daño...")
    for img_path in tqdm(clean_list, desc="Clean Pipeline"):
        try:
            # 1. Obtener el polígono del mapa cargado en memoria
            # Si no existe, devuelve None (y el generador hará fallback a imagen completa o vacía)
            car_polygon = polygons_map.get(img_path.name)
            
            if car_polygon is None:
                # Opcional: Warning si esperábamos que todas tuvieran polígono
                # print(f"⚠️ No hay polígono para {img_path.name}, usando imagen completa.")
                pass

            # 2. Generar crops filtrados por ese polígono
            crops = grid_gen.generate_crops(img_path, car_polygon=car_polygon)
            
            for i, crop_data in enumerate(crops):
                crop_filename = f"{img_path.stem}_grid_{crop_data['grid_x']}_{crop_data['grid_y']}.jpg"
                save_path = output_base / "clean" / crop_filename
                
                cv2.imwrite(str(save_path), crop_data['crop'])
                
                metadata_list.append({
                    'crop_id': crop_filename,
                    'crop_path': str(save_path),
                    'source_image': img_path.name,
                    'has_damage': False,
                    'vehicle_zone': get_vehicle_zone(img_path.name),
                    'generation_method': 'grid_masked_polygon',
                    'vehicle_ratio': crop_data.get('vehicle_ratio', 1.0),
                    'grid_pos': {'x': crop_data['grid_x'], 'y': crop_data['grid_y']}
                })
                
        except Exception as e:
            print(f"❌ Error en {img_path.name}: {e}")

    # --- FINALIZAR ---
    meta_path = output_base.parent / "metadata_crops_phase1.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
        
    print(f"\n✅ Fase 1 Completada.")
    print(f"   - Total crops: {len(metadata_list)}")
    print(f"   - Metadata: {meta_path}")

if __name__ == "__main__":
    main()
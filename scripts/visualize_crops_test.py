#!/usr/bin/env python3
# scripts/visualize_crops_test.py

import sys
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Añadir raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.preprocessing.clustered_crop_generator_optimized import ClusteredCropGeneratorOptimized
from src.core.preprocessing.grid_crop_generator import GridCropGenerator

def visualize_and_save(image_path, crops, output_folder, title="Crops", color='red'):
    """
    1. Dibuja los bboxes de los crops sobre la imagen original.
    2. Guarda los crops individuales en disco.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error leyendo imagen: {image_path}")
        return
    
    # Convertir a RGB para matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # --- 1. VISUALIZACIÓN EN PANTALLA ---
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    ax = plt.gca()
    
    print(f"\n📸 Generando {len(crops)} crops para: {Path(image_path).name}")
    
    # Crear directorio de salida
    save_dir = Path(output_folder) / Path(image_path).stem
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, crop_data in enumerate(crops):
        # --- LÓGICA DE COORDENADAS ADAPTATIVA ---
        
        # CASO A: Clustered Crop (Smart Tiling V2)
        if 'bbox_crop' in crop_data:
            x1, y1, x2, y2 = crop_data['bbox_crop']
            w, h = x2 - x1, y2 - y1
            
            # Etiqueta enriquecida
            cluster_id = crop_data.get('cluster_id', i)
            if crop_data.get('is_tiled', False):
                tile_id = crop_data.get('tile_id', 0)
                label = f"C{cluster_id}-T{tile_id}" # Cluster X, Tile Y
            else:
                label = f"Cluster {cluster_id}"

        # CASO B: Grid Crop (Clean)
        elif 'grid_x' in crop_data:
            x1 = crop_data['grid_x']
            y1 = crop_data['grid_y']
            h_crop, w_crop = crop_data['crop'].shape[:2]
            w, h = w_crop, h_crop
            label = f"Grid {i}"
            
        # CASO C: Legacy (Por si acaso)
        elif 'bbox_original' in crop_data:
            x1, y1, x2, y2 = crop_data['bbox_original']
            w, h = x2 - x1, y2 - y1
            label = f"Legacy {i}"
            
        else:
            print(f"⚠️ Crop {i} con metadata desconocida: {crop_data.keys()}")
            continue
        
        # Dibujar rectángulo
        rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label, color=color, fontsize=8, weight='bold', backgroundcolor='white')
        
        # --- 2. GUARDADO EN DISCO ---
        # Nombre descriptivo
        if 'bbox_crop' in crop_data and crop_data.get('is_tiled', False):
             tile_id = crop_data.get('tile_id', 0)
             cluster_id = crop_data.get('cluster_id', 0)
             crop_name = f"crop_{i:03d}_C{cluster_id}_T{tile_id}.jpg"
        else:
             crop_name = f"crop_{i:03d}.jpg"
             
        cv2.imwrite(str(save_dir / crop_name), crop_data['crop'])
    
    plt.title(f"{title} - {len(crops)} recortes generados")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print(f"💾 Crops guardados en: {save_dir}")

def main():
    # --- 1. CONFIGURACIÓN GENERAL ---
    base_dir = Path("data/raw/masked_dataset_split_2")  # Asegúrate que esta ruta es correcta
    poligonos_path = base_dir / "poligonos.json"        # Para las clean
    output_root = Path("output-prueba")
    
    # --- 2. INPUTS MANUALES (¡CAMBIA ESTO PARA CADA PRUEBA!) ---
    
    # IMAGEN CON DAÑO (Debe tener su _labelDANO...json al lado)
    file_damaged = "zona1_ok_2_3_1563173327452_zona_8_imageDANO_original.jpg"
    
    # IMAGEN SIN DAÑO (Debe estar en poligonos.json)
    file_clean = "zona1_ko_2_3_1554817134014_zona_4_imageDANO_original.jpg" 
    
    
    # --- EJECUCIÓN PRUEBA 1: IMAGEN CON DAÑO ---
    print(f"\n{'='*60}\n🧪 PRUEBA 1: MANUAL DAÑO (CLUSTERED - SMART TILING)\n{'='*60}")
    
    if file_damaged:
        img_path = base_dir / file_damaged
        # Construimos la ruta del JSON de defectos esperado
        # Intentamos primero el nombre estandar labelDANO
        json_path = img_path.parent / img_path.name.replace('_imageDANO_original.jpg', '_labelDANO_modificado.json')
        
        # Si no existe, probamos naming alternativo
        if not json_path.exists():
             json_path = img_path.with_name(img_path.stem + "_labelDANO_modificado.json")

        if img_path.exists() and json_path.exists():
            print(f"✅ Procesando: {file_damaged}")
            # Usamos 336px como acordamos
            generator = ClusteredCropGeneratorOptimized(target_size=336)
            crops = generator.generate_crops(img_path, json_path)
            
            visualize_and_save(
                img_path, 
                crops, 
                output_root / "damaged_manual", 
                title=f"Damaged (Tiling): {file_damaged}", 
                color='red'
            )
        else:
            print(f"❌ NO SE PUEDE PROCESAR COMO DAÑO:")
            print(f"   - Imagen existe: {img_path.exists()}")
            print(f"   - JSON defectos existe: {json_path.exists()}")
            print(f"   - Ruta JSON buscada: {json_path}")
    else:
        print("⏭️  Salto prueba de daño (variable vacía)")


    # --- EJECUCIÓN PRUEBA 2: IMAGEN SIN DAÑO (CLEAN) ---
    print(f"\n{'='*60}\n🧪 PRUEBA 2: MANUAL CLEAN (GRID MASKED)\n{'='*60}")
    
    if file_clean:
        img_path = base_dir / file_clean
        
        if img_path.exists():
            print(f"✅ Procesando: {file_clean}")
            
            # Buscar polígono
            polygon = None
            if poligonos_path.exists():
                with open(poligonos_path) as f:
                    poly_data = json.load(f)
                    for item in poly_data:
                        if item['filename'] == file_clean:
                            polygon = item['polygon']
                            print("   ✅ Polígono encontrado en JSON maestro.")
                            break
            
            if polygon is None:
                print("   ⚠️  No se encontró polígono (se usará Grid sobre toda la imagen/fallback).")

            # Generar crops
            grid_gen = GridCropGenerator(crop_size=336, overlap=0.30, min_vehicle_ratio=0.40)
            crops = grid_gen.generate_crops(img_path, car_polygon=polygon)
            
            visualize_and_save(
                img_path, 
                crops, 
                output_root / "clean_manual", 
                title=f"Clean (Masked): {file_clean}", 
                color='lime'
            )
        else:
            print(f"❌ Error: No existe la imagen: {img_path}")
    else:
        print("⏭️  Salto prueba clean (variable vacía)")

if __name__ == "__main__":
    main()
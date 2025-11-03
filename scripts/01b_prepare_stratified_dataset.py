# scripts/01b_prepare_stratified_dataset.py

import json
import shutil
from pathlib import Path
from collections import Counter
import numpy as np

def analyze_and_select_stratified(
    source_dir: Path,
    output_base_dir: Path,
    samples_per_subset: int = 20
):
    """
    Selecciona 3 subsets estratificados por densidad de defectos
    
    Args:
        source_dir: Directorio con dataset original
        output_base_dir: Directorio base de salida
        samples_per_subset: Imágenes por subset (default: 20)
    """
    
    # 1. Escanear dataset completo y calcular densidad de defectos
    json_files = list(source_dir.glob("*labelDANO_modificado.json"))
    print(f"📊 Analizando {len(json_files)} imágenes del dataset completo...")
    
    image_stats = []
    
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        defect_count = len([s for s in data['shapes'] if s['shape_type'] == 'polygon'])
        
        # Solo considerar imágenes con al menos 1 defecto
        if defect_count < 1:
            continue
        
        image_stats.append({
            'json_path': json_path,
            'image_path': source_dir / data['imagePath'],
            'defect_count': defect_count,
            'image_name': data['imagePath']
        })
    
    # Ordenar por número de defectos
    image_stats.sort(key=lambda x: x['defect_count'])
    
    defect_counts = [img['defect_count'] for img in image_stats]
    mean_defects = np.mean(defect_counts)
    median_defects = np.median(defect_counts)
    
    print(f"\n📈 Estadísticas del dataset:")
    print(f"   - Total imágenes válidas: {len(image_stats)}")
    print(f"   - Defectos por imagen: min={min(defect_counts)}, max={max(defect_counts)}")
    print(f"   - Media: {mean_defects:.1f}, Mediana: {median_defects:.0f}")
    
    # 2. Seleccionar 3 subsets
    subsets = {
        'high_density': image_stats[-samples_per_subset:],  # Últimas 20 (más defectos)
        'low_density': image_stats[:samples_per_subset],     # Primeras 20 (menos defectos)
    }
    
    # Medium: seleccionar alrededor de la mediana
    median_idx = len(image_stats) // 2
    start_idx = max(0, median_idx - samples_per_subset // 2)
    subsets['medium_density'] = image_stats[start_idx:start_idx + samples_per_subset]
    
    # 3. Copiar archivos y crear manifests
    print(f"\n📦 Creando subsets...\n")
    
    for subset_name, images in subsets.items():
        output_dir = output_base_dir / subset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = []
        defects_list = []
        
        for idx, img_meta in enumerate(images):
            # Copiar imagen y JSON
            img_dst = output_dir / img_meta['image_path'].name
            json_dst = output_dir / img_meta['json_path'].name
            
            shutil.copy2(img_meta['image_path'], img_dst)
            shutil.copy2(img_meta['json_path'], json_dst)
            
            manifest.append({
                'id': idx,
                'image': img_meta['image_name'],
                'json': img_meta['json_path'].name,
                'defect_count': img_meta['defect_count']
            })
            
            defects_list.append(img_meta['defect_count'])
        
        # Guardar manifest
        manifest_path = output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Stats del subset
        print(f"✅ {subset_name.upper()}:")
        print(f"   - Imágenes: {len(images)}")
        print(f"   - Defectos: min={min(defects_list)}, max={max(defects_list)}, mean={np.mean(defects_list):.1f}")
        print(f"   - Guardado en: {output_dir}\n")
    
    print("✨ Proceso completado!")


if __name__ == "__main__":
    SOURCE_DIR = Path("/home/carlos/Escritorio/Proyectos-Minsait/Mapfre/carpetas-datos/jsons_segmentacion_jsonsfinales")
    OUTPUT_BASE_DIR = Path("data/raw/stratified_subsets")
    
    analyze_and_select_stratified(
        source_dir=SOURCE_DIR,
        output_base_dir=OUTPUT_BASE_DIR,
        samples_per_subset=20
    )
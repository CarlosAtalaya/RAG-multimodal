#!/usr/bin/env python3
# scripts/01d_prepare_dataset_split_8020.py

"""
📊 TRAIN/TEST SPLIT ESTRATIFICADO 80/20

Mejoras vs 01c:
- 80/20 en lugar de 30/70 (más train para índice robusto)
- Estratifica por DENSIDAD + ZONA
- Elimina label "9" (solo 1 muestra)
- Metadata enriquecida con info de zonas
"""

from pathlib import Path
import json
import random
import shutil
from collections import defaultdict, Counter
from typing import List, Dict
import numpy as np
import yaml

# Cargar config de zonas
with open("config/crop_strategy_config.yaml") as f:
    CONFIG = yaml.safe_load(f)

VEHICLE_ZONES = CONFIG['vehicle_zones']

def analyze_image_defects(json_path: Path) -> Dict:
    """Analiza defectos con info de zona del vehículo"""
    with open(json_path) as f:
        data = json.load(f)
    
    # Extraer zona del vehículo del nombre
    vehicle_zone = extract_vehicle_zone(data['imagePath'])
    
    # Contar defectos por tipo
    defect_counts = Counter()
    valid_defects = []
    
    for shape in data['shapes']:
        if shape['shape_type'] != 'polygon':
            continue
        
        label = shape['label']
        
        # Filtrar label "9" (anómalo)
        if label == "9":
            continue
        
        defect_counts[label] += 1
        valid_defects.append(shape)
    
    return {
        'json_path': json_path,
        'image_path': json_path.parent / data['imagePath'],
        'vehicle_zone': vehicle_zone,
        'zone_description': VEHICLE_ZONES.get(vehicle_zone, {}).get('description', 'unknown'),
        'zone_area': VEHICLE_ZONES.get(vehicle_zone, {}).get('area', 'unknown'),
        'total_defects': sum(defect_counts.values()),
        'defect_distribution': dict(defect_counts),
        'image_size': [data['imageWidth'], data['imageHeight']],
        'valid_shapes': valid_defects
    }


def extract_vehicle_zone(image_path: str) -> str:
    """Extrae zona del vehículo del nombre"""
    # Formato: zona1_ko_2_3_1554114337244_zona_5_imageDANO_original.jpg
    parts = image_path.split('_')
    for i, part in enumerate(parts):
        if part == 'zona' and i + 1 < len(parts):
            zone_num = parts[i + 1].split('.')[0]  # Quitar extensión
            if zone_num.isdigit():
                return zone_num
    return "unknown"


def stratified_split_8020(
    source_dir: Path,
    output_dir: Path,
    train_size: float = 0.80,
    random_seed: int = 42
):
    """
    Split estratificado 80/20 balanceando por:
    1. Densidad de defectos (low/medium/high)
    2. Zona del vehículo (1-10)
    """
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"\n{'='*70}")
    print(f"📊 TRAIN/TEST SPLIT ESTRATIFICADO 80/20")
    print(f"{'='*70}\n")
    
    # 1. Escanear dataset
    print("🔍 Escaneando dataset...")
    json_files = list(source_dir.glob("*labelDANO_modificado.json"))
    print(f"   ✅ {len(json_files)} archivos encontrados\n")
    
    # 2. Analizar cada imagen
    print("📊 Analizando defectos...")
    image_data = []
    
    for json_path in json_files:
        try:
            meta = analyze_image_defects(json_path)
            if meta['total_defects'] > 0:  # Solo con defectos
                image_data.append(meta)
        except Exception as e:
            print(f"   ⚠️  Error en {json_path.name}: {e}")
    
    print(f"   ✅ {len(image_data)} imágenes válidas\n")
    
    # 3. Calcular cuartiles de densidad
    print("📦 Calculando densidad...")
    defect_counts = [img['total_defects'] for img in image_data]
    q33 = np.percentile(defect_counts, 33)
    q66 = np.percentile(defect_counts, 66)
    
    print(f"   - Q33: {q33:.0f} defectos")
    print(f"   - Q66: {q66:.0f} defectos\n")
    
    # 4. Agrupar por densidad Y zona
    print("🗂️  Agrupando por densidad y zona...")
    
    groups = defaultdict(list)
    
    for img in image_data:
        n_defects = img['total_defects']
        vehicle_zone = img['vehicle_zone']
        
        # Determinar densidad
        if n_defects <= q33:
            density = 'low'
        elif n_defects <= q66:
            density = 'medium'
        else:
            density = 'high'
        
        # Clave: densidad_zona
        key = f"{density}_{vehicle_zone}"
        groups[key].append(img)
    
    print(f"   ✅ {len(groups)} grupos creados\n")
    
    # 5. Split estratificado dentro de cada grupo
    print("✂️  Dividiendo en train/test...")
    
    train_set = []
    test_set = []
    
    for group_key, images in sorted(groups.items()):
        n_images = len(images)
        n_train = int(n_images * train_size)
        
        # Shuffle
        random.shuffle(images)
        
        # Split
        train_images = images[:n_train]
        test_images = images[n_train:]
        
        train_set.extend(train_images)
        test_set.extend(test_images)
        
        if n_images >= 5:  # Solo mostrar grupos significativos
            print(f"   {group_key:20} | Total: {n_images:4} | Train: {len(train_images):4} | Test: {len(test_images):4}")
    
    print(f"\n   {'='*60}")
    print(f"   {'TOTAL':20} | Total: {len(image_data):4} | Train: {len(train_set):4} | Test: {len(test_set):4}")
    print(f"   {'='*60}\n")
    
    # 6. Verificar distribución
    print("🔍 Verificando distribución...\n")
    
    train_types = Counter()
    test_types = Counter()
    
    for img in train_set:
        for dtype, count in img['defect_distribution'].items():
            train_types[dtype] += count
    
    for img in test_set:
        for dtype, count in img['defect_distribution'].items():
            test_types[dtype] += count
    
    print("   Train vs Test por tipo:")
    print(f"   {'Tipo':15} | {'Train':>10} | {'Test':>10} | {'% Test':>8}")
    print(f"   {'-'*50}")
    
    all_types = set(train_types.keys()) | set(test_types.keys())
    for dtype in sorted(all_types, key=lambda x: train_types[x] + test_types[x], reverse=True):
        train_count = train_types[dtype]
        test_count = test_types[dtype]
        total = train_count + test_count
        test_pct = (test_count / total * 100) if total > 0 else 0
        
        print(f"   {dtype:15} | {train_count:10} | {test_count:10} | {test_pct:7.1f}%")
    
    print()
    
    # 7. Copiar archivos
    print("📁 Copiando archivos...")
    
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    def copy_dataset(images: List[Dict], dest_dir: Path, set_name: str):
        manifest = []
        
        for idx, img in enumerate(images):
            # Copiar imagen
            img_dst = dest_dir / img['image_path'].name
            shutil.copy2(img['image_path'], img_dst)
            
            # Copiar JSON
            json_dst = dest_dir / img['json_path'].name
            shutil.copy2(img['json_path'], json_dst)
            
            manifest.append({
                'id': idx,
                'image': img['image_path'].name,
                'json': img['json_path'].name,
                'vehicle_zone': img['vehicle_zone'],
                'zone_description': img['zone_description'],
                'zone_area': img['zone_area'],
                'total_defects': img['total_defects'],
                'defect_distribution': img['defect_distribution'],
                'image_size': img['image_size']
            })
        
        # Guardar manifest
        manifest_path = dest_dir / f'{set_name}_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"   ✅ {set_name.upper()}: {len(images)} imágenes → {dest_dir}")
        return manifest_path
    
    train_manifest = copy_dataset(train_set, train_dir, 'train')
    test_manifest = copy_dataset(test_set, test_dir, 'test')
    
    # 8. Guardar info del split
    split_info = {
        'random_seed': random_seed,
        'train_size': train_size,
        'test_size': 1 - train_size,
        'total_images': len(image_data),
        'train_images': len(train_set),
        'test_images': len(test_set),
        'train_manifest': str(train_manifest),
        'test_manifest': str(test_manifest),
        'defect_distribution': {
            'train': dict(train_types),
            'test': dict(test_types)
        },
        'density_thresholds': {
            'q33': float(q33),
            'q66': float(q66)
        }
    }
    
    split_info_path = output_dir / 'split_info.json'
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n💾 Info del split: {split_info_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ SPLIT COMPLETADO")
    print(f"{'='*70}")
    print(f"Train: {len(train_set)} imágenes ({train_size*100:.0f}%)")
    print(f"Test:  {len(test_set)} imágenes ({(1-train_size)*100:.0f}%)")
    print(f"{'='*70}\n")
    
    return train_manifest, test_manifest, split_info_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source',
        type=Path,
        default=Path("/home/carlos/Escritorio/Proyectos-Minsait/Mapfre/carpetas-datos/jsons_segmentacion_jsonsfinales"),
        help='Directorio con dataset original'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/raw/train_test_split_8020"),
        help='Directorio de salida'
    )
    parser.add_argument(
        '--train-size',
        type=float,
        default=0.80,
        help='Proporción del train set'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )
    
    args = parser.parse_args()
    
    try:
        stratified_split_8020(
            source_dir=args.source,
            output_dir=args.output,
            train_size=args.train_size,
            random_seed=args.seed
        )
        
        print("✨ ¡Split completado!")
        print("\n📌 Próximo paso:")
        print("   python scripts/03c_generate_fullimage_embeddings.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
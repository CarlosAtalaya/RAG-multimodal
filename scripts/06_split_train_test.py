#!/usr/bin/env python3
# scripts/06_split_train_test.py

"""
📊 TRAIN/TEST SPLIT ESTRATIFICADO

Divide el dataset en:
- Train Set (80%): Para generar embeddings e índice FAISS
- Test Set (20%): Para evaluación RAG (NUNCA visto en training)

Estratificación:
- Balancea por tipo de daño
- Balancea por densidad de defectos
- Asegura que ambos sets sean representativos
"""

from pathlib import Path
import json
import random
import shutil
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np

def analyze_image_defects(json_path: Path) -> Dict:
    """Analiza defectos de una imagen"""
    with open(json_path) as f:
        data = json.load(f)
    
    defect_counts = Counter([
        shape['label'] for shape in data['shapes']
        if shape['shape_type'] == 'polygon'
    ])
    
    return {
        'json_path': json_path,
        'image_path': json_path.parent / data['imagePath'],
        'total_defects': sum(defect_counts.values()),
        'defect_distribution': dict(defect_counts),
        'dominant_type': defect_counts.most_common(1)[0][0] if defect_counts else None
    }


def stratified_split(
    source_dir: Path,
    output_dir: Path,
    test_size: float = 0.20,
    random_seed: int = 42
):
    """
    Split estratificado del dataset
    
    Args:
        source_dir: Directorio con imágenes y JSONs
        output_dir: Directorio de salida
        test_size: Proporción del test set (default: 20%)
        random_seed: Semilla para reproducibilidad
    """
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"\n{'='*70}")
    print(f"📊 TRAIN/TEST SPLIT ESTRATIFICADO")
    print(f"{'='*70}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Test size: {test_size*100:.0f}%")
    print(f"Random seed: {random_seed}")
    print(f"{'='*70}\n")
    
    # 1. Escanear todas las imágenes
    print("🔍 Escaneando dataset...")
    json_files = list(source_dir.glob("*labelDANO_modificado.json"))
    
    if not json_files:
        raise ValueError(f"No se encontraron JSONs en {source_dir}")
    
    print(f"   ✅ {len(json_files)} imágenes encontradas\n")
    
    # 2. Analizar cada imagen
    print("📊 Analizando defectos...")
    image_data = []
    
    for json_path in json_files:
        try:
            meta = analyze_image_defects(json_path)
            if meta['total_defects'] > 0:  # Solo imágenes con defectos
                image_data.append(meta)
        except Exception as e:
            print(f"   ⚠️  Error en {json_path.name}: {e}")
    
    print(f"   ✅ {len(image_data)} imágenes válidas\n")
    
    # 3. Estratificar por densidad de defectos
    print("📦 Estratificando por densidad de defectos...")
    
    # Calcular cuartiles de densidad
    defect_counts = [img['total_defects'] for img in image_data]
    q25 = np.percentile(defect_counts, 25)
    q50 = np.percentile(defect_counts, 50)
    q75 = np.percentile(defect_counts, 75)
    
    print(f"   - Q25: {q25:.0f} defectos")
    print(f"   - Q50: {q50:.0f} defectos")
    print(f"   - Q75: {q75:.0f} defectos\n")
    
    # Agrupar por densidad
    density_groups = {
        'very_low': [],
        'low': [],
        'medium': [],
        'high': []
    }
    
    for img in image_data:
        n_defects = img['total_defects']
        if n_defects <= q25:
            density_groups['very_low'].append(img)
        elif n_defects <= q50:
            density_groups['low'].append(img)
        elif n_defects <= q75:
            density_groups['medium'].append(img)
        else:
            density_groups['high'].append(img)
    
    # 4. Split estratificado dentro de cada grupo
    print("✂️  Dividiendo en train/test...")
    
    train_set = []
    test_set = []
    
    for density, images in density_groups.items():
        n_images = len(images)
        n_test = int(n_images * test_size)
        
        # Shuffle
        random.shuffle(images)
        
        # Split
        test_images = images[:n_test]
        train_images = images[n_test:]
        
        train_set.extend(train_images)
        test_set.extend(test_images)
        
        print(f"   {density:12} | Total: {n_images:4} | Train: {len(train_images):4} | Test: {n_test:4}")
    
    print(f"\n   {'='*60}")
    print(f"   {'TOTAL':12} | Total: {len(image_data):4} | Train: {len(train_set):4} | Test: {len(test_set):4}")
    print(f"   {'='*60}\n")
    
    # 5. Verificar distribución de tipos de daño
    print("🔍 Verificando distribución de tipos de daño...")
    
    train_types = Counter()
    test_types = Counter()
    
    for img in train_set:
        for dtype, count in img['defect_distribution'].items():
            train_types[dtype] += count
    
    for img in test_set:
        for dtype, count in img['defect_distribution'].items():
            test_types[dtype] += count
    
    print("\n   Train vs Test por tipo:")
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
    
    # 6. Copiar archivos
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
                'total_defects': img['total_defects'],
                'defect_distribution': img['defect_distribution']
            })
        
        # Guardar manifest
        manifest_path = dest_dir / f'{set_name}_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"   ✅ {set_name.upper()}: {len(images)} imágenes → {dest_dir}")
        return manifest_path
    
    train_manifest = copy_dataset(train_set, train_dir, 'train')
    test_manifest = copy_dataset(test_set, test_dir, 'test')
    
    # 7. Guardar info del split
    split_info = {
        'random_seed': random_seed,
        'test_size': test_size,
        'total_images': len(image_data),
        'train_images': len(train_set),
        'test_images': len(test_set),
        'train_manifest': str(train_manifest),
        'test_manifest': str(test_manifest),
        'defect_distribution': {
            'train': dict(train_types),
            'test': dict(test_types)
        }
    }
    
    split_info_path = output_dir / 'split_info.json'
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n💾 Info del split: {split_info_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ SPLIT COMPLETADO")
    print(f"{'='*70}")
    print(f"Train: {len(train_set)} imágenes ({(1-test_size)*100:.0f}%)")
    print(f"Test:  {len(test_set)} imágenes ({test_size*100:.0f}%)")
    print(f"{'='*70}\n")
    
    return train_manifest, test_manifest, split_info_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Split dataset en train/test estratificado'
    )
    parser.add_argument(
        '--source',
        type=Path,
        default=Path("/home/carlos/Escritorio/Proyectos-Minsait/Mapfre/carpetas-datos/jsons_segmentacion_jsonsfinales"),
        help='Directorio con dataset original'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/raw/train_test_split"),
        help='Directorio de salida'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.70,
        help='Proporción del test set (default: 0.70 = 70%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla aleatoria para reproducibilidad'
    )
    
    args = parser.parse_args()
    
    try:
        train_manifest, test_manifest, split_info = stratified_split(
            source_dir=args.source,
            output_dir=args.output,
            test_size=args.test_size,
            random_seed=args.seed
        )
        
        print("✨ ¡Listo! Próximos pasos:")
        print("\n1. Generar embeddings SOLO del train set:")
        print(f"   python scripts/02_generate_clustered_crops.py \\")
        print(f"       --dataset data/raw/train_test_split/train")
        print()
        print("2. Evaluar con test set:")
        print(f"   python scripts/07_evaluate_rag_end_to_end.py \\")
        print(f"       --test-set data/raw/train_test_split/test")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
# scripts/03c_generate_fullimage_embeddings.py

"""
🧠 GENERACIÓN DE EMBEDDINGS - IMÁGENES COMPLETAS + METADATA ENRIQUECIDA

Características:
1. Embedding de IMAGEN COMPLETA (no crops)
2. Metadata enriquecida:
   - Lista de defectos con posiciones
   - Zonas del vehículo afectadas
   - Descripción textual automática
   - Bboxes de todos los defectos
3. Normalización consistente con DINOv3
"""

from pathlib import Path
import json
import numpy as np
import sys
from datetime import datetime
from typing import List, Dict
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embeddings.dinov3_vitl_embedder import DINOv3ViTLEmbedder

# Cargar config
with open("config/crop_strategy_config.yaml") as f:
    CONFIG = yaml.safe_load(f)

VEHICLE_ZONES = CONFIG['vehicle_zones']

DAMAGE_TYPES = {
    "1": "surface_scratch",
    "2": "dent",
    "3": "paint_peeling",
    "4": "deep_scratch",
    "5": "crack",
    "6": "missing_part",
    "7": "missing_accessory",
    "8": "misaligned_part"
}


def build_defect_description(defects: List[Dict], vehicle_zone: str) -> str:
    """
    Construye descripción textual de defectos para metadata
    
    Ejemplo: "3 surface scratches and 1 dent on hood center (frontal area)"
    """
    from collections import Counter
    
    # Contar por tipo
    type_counts = Counter([d['damage_type'] for d in defects])
    
    # Construir descripción
    parts = []
    for dtype, count in type_counts.most_common():
        plural = "s" if count > 1 else ""
        parts.append(f"{count} {dtype}{plural}")
    
    damage_desc = " and ".join(parts) if parts else "no defects"
    
    zone_info = VEHICLE_ZONES.get(vehicle_zone, {})
    zone_desc = zone_info.get('description', 'unknown area')
    zone_area = zone_info.get('area', 'unknown')
    
    return f"{damage_desc} on {zone_desc} ({zone_area})"


def calculate_spatial_distribution(defects: List[Dict], img_w: int, img_h: int) -> Dict:
    """
    Calcula distribución espacial de defectos en grilla 3×3
    
    Returns:
        Dict con conteo por zona: {top_left: 2, middle_center: 5, ...}
    """
    from collections import defaultdict
    
    spatial_counts = defaultdict(int)
    
    for defect in defects:
        bbox = defect['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        rel_x = center_x / img_w
        rel_y = center_y / img_h
        
        zone_x = "left" if rel_x < 0.33 else "center" if rel_x < 0.67 else "right"
        zone_y = "top" if rel_y < 0.33 else "middle" if rel_y < 0.67 else "bottom"
        
        zone_key = f"{zone_y}_{zone_x}"
        spatial_counts[zone_key] += 1
    
    return dict(spatial_counts)


def generate_fullimage_embeddings(
    train_dir: Path,
    output_dir: Path,
    batch_size: int = 8
):
    """
    Pipeline completo de generación de embeddings para imágenes completas
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 GENERACIÓN DE EMBEDDINGS - IMÁGENES COMPLETAS")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # 1. Cargar manifest
    print("📄 Cargando manifest...")
    manifest_path = train_dir / "train_manifest.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"   ✅ {len(manifest)} imágenes en train set\n")
    
    # 2. Inicializar embedder
    print("🤖 Inicializando DINOv3-ViT-L/16...")
    embedder = DINOv3ViTLEmbedder(use_bfloat16=True)
    print()
    
    # 3. Preparar datos
    print("📦 Preparando datos...")
    
    image_paths = []
    enriched_metadata = []
    
    for item in manifest:
        json_path = train_dir / item['json']
        image_path = train_dir / item['image']
        
        if not image_path.exists():
            print(f"   ⚠️  Imagen no encontrada: {image_path.name}")
            continue
        
        # Cargar JSON para extraer defectos
        with open(json_path) as f:
            json_data = json.load(f)
        
        # Procesar defectos
        defects = []
        for shape in json_data['shapes']:
            if shape['shape_type'] != 'polygon':
                continue
            
            label = shape['label']
            if label == "9":  # Filtrar anómalo
                continue
            
            polygon = np.array(shape['points'], dtype=np.int32)
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            
            defect = {
                'damage_type': DAMAGE_TYPES.get(label, 'unknown'),
                'damage_label': label,
                'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                'bbox_area': int((x_max - x_min) * (y_max - y_min)),
                'polygon_coords': polygon.tolist()
            }
            
            defects.append(defect)
        
        if not defects:
            continue
        
        # Construir metadata enriquecida
        img_w, img_h = item['image_size']
        vehicle_zone = item['vehicle_zone']
        
        meta = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'image_size': [img_w, img_h],
            
            # Info de vehículo
            'vehicle_zone': vehicle_zone,
            'zone_description': item['zone_description'],
            'zone_area': item['zone_area'],
            
            # Defectos agregados
            'total_defects': len(defects),
            'defect_types': list(set([d['damage_type'] for d in defects])),
            'defect_distribution': item['defect_distribution'],
            'defects': defects,
            
            # Descripción textual
            'description': build_defect_description(defects, vehicle_zone),
            
            # Distribución espacial
            'spatial_distribution': calculate_spatial_distribution(defects, img_w, img_h),
            
            # Info de bboxes
            'bboxes': [d['bbox'] for d in defects],
            'total_bbox_area': sum([d['bbox_area'] for d in defects]),
            
            # Estadísticas
            'avg_defect_size': np.mean([d['bbox_area'] for d in defects]),
            'max_defect_size': max([d['bbox_area'] for d in defects]),
            'min_defect_size': min([d['bbox_area'] for d in defects])
        }
        
        image_paths.append(image_path)
        enriched_metadata.append(meta)
    
    valid_images = len(image_paths)
    print(f"   ✅ {valid_images} imágenes válidas para procesar\n")
    
    # 4. Generar embeddings
    print(f"🧠 Generando embeddings (batch_size={batch_size})...\n")
    
    embeddings = embedder.generate_batch_embeddings(
        image_paths,
        batch_size=batch_size,
        normalize=True,
        show_progress=True
    )
    
    print(f"\n{'='*70}")
    print(f"✅ EMBEDDINGS GENERADOS")
    print(f"{'='*70}")
    print(f"Shape: {embeddings.shape}")
    print(f"Dimensión: {embeddings.shape[1]}")
    print(f"{'='*70}\n")
    
    # 5. Estadísticas
    print("📊 Estadísticas de embeddings:")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"   - Norma promedio: {norms.mean():.4f}")
    print(f"   - Norma std: {norms.std():.4f}\n")
    
    # 6. Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("💾 Guardando archivos...")
    
    # Embeddings
    embeddings_path = output_dir / "embeddings_fullimages_dinov3.npy"
    np.save(embeddings_path, embeddings)
    print(f"   ✅ Embeddings: {embeddings_path}")
    
    # Metadata
    for i, meta in enumerate(enriched_metadata):
        meta['embedding_index'] = i
        meta['embedding_model'] = 'dinov3-vitl16-fullimage'
        meta['embedding_dim'] = int(embeddings.shape[1])
        meta['embedding_norm'] = float(norms[i])
    
    metadata_path = output_dir / "metadata_fullimages.json"
    with open(metadata_path, 'w') as f:
        json.dump(enriched_metadata, f, indent=2)
    print(f"   ✅ Metadata: {metadata_path}\n")
    
    # 7. Info del proceso
    process_info = {
        'timestamp': datetime.now().isoformat(),
        'model': embedder.get_model_info(),
        'dataset': {
            'total_images': len(manifest),
            'valid_images': valid_images,
            'train_dir': str(train_dir)
        },
        'embeddings': {
            'shape': list(embeddings.shape),
            'dtype': str(embeddings.dtype),
            'norm_mean': float(norms.mean()),
            'norm_std': float(norms.std())
        },
        'output_files': {
            'embeddings': str(embeddings_path),
            'metadata': str(metadata_path)
        }
    }
    
    info_path = output_dir / "generation_info.json"
    with open(info_path, 'w') as f:
        json.dump(process_info, f, indent=2)
    print(f"📋 Info del proceso: {info_path}\n")
    
    print(f"{'='*70}")
    print(f"✨ PROCESO COMPLETADO")
    print(f"{'='*70}\n")
    
    return embeddings, enriched_metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-dir',
        type=Path,
        default=Path("data/raw/train_test_split_8020/train"),
        help='Directorio con train set'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/processed/embeddings/fullimages_dinov3"),
        help='Directorio de salida'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Tamaño del batch'
    )
    
    args = parser.parse_args()
    
    try:
        generate_fullimage_embeddings(
            train_dir=args.train_dir,
            output_dir=args.output,
            batch_size=args.batch_size
        )
        
        print("📌 Próximo paso:")
        print("   python scripts/04_build_faiss_index.py \\")
        print(f"       --embeddings {args.output}/embeddings_fullimages_dinov3.npy \\")
        print(f"       --metadata {args.output}/metadata_fullimages.json")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
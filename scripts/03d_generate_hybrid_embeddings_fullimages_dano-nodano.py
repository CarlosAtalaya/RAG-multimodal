#!/usr/bin/env python3
# scripts/03d_generate_hybrid_embeddings_fullimages_dano-nodano.py

"""
🌟 GENERACIÓN DE EMBEDDINGS HÍBRIDOS - IMÁGENES CON Y SIN DAÑO

Procesa:
- Imágenes CON daño (*_labelDANO_modificado.json) → metadata rica con defectos
- Imágenes SIN daño (*_imageDANO_original.json) → metadata sin defectos

Output:
- Embeddings híbridos (visual + text) de 1408 dims
- Metadata unificada para ambos tipos
"""

from pathlib import Path
import json
import numpy as np
import sys
from datetime import datetime
from typing import List, Dict, Tuple
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embeddings.multimodal_embedder_v2 import MultimodalEmbedderV2

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


def extract_vehicle_zone(image_name: str) -> str:
    """Extrae zona del vehículo del nombre"""
    parts = image_name.split('_')
    for i, part in enumerate(parts):
        if part == 'zona' and i + 1 < len(parts):
            zone_num = parts[i + 1].split('.')[0]
            if zone_num.isdigit():
                return zone_num
    return "unknown"


def is_damage_image(json_path: Path) -> bool:
    """Determina si es imagen con daño por el nombre del JSON"""
    return json_path.name.endswith('_labelDANO_modificado.json')


def process_damage_image(image_path: Path, json_path: Path) -> Dict:
    """Procesa imagen CON daño (con segmentaciones)"""
    
    with open(json_path) as f:
        json_data = json.load(f)
    
    img_w, img_h = json_data.get('imageWidth', 0), json_data.get('imageHeight', 0)
    vehicle_zone = extract_vehicle_zone(image_path.name)
    
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
        # JSON con daño pero sin shapes válidos → tratar como sin daño
        return process_no_damage_image(image_path, json_path)
    
    # Construir metadata rica
    from collections import Counter
    defect_types = [d['damage_type'] for d in defects]
    defect_distribution = dict(Counter([d['damage_label'] for d in defects]))
    
    # Distribución espacial
    spatial_dist = calculate_spatial_distribution(defects, img_w, img_h)
    
    # Descripción textual
    description = build_damage_description(defects, vehicle_zone)
    
    metadata = {
        'image_path': str(image_path),
        'image_name': image_path.name,
        'image_size': [img_w, img_h],
        'has_damage': True,  # ← FLAG CRÍTICO
        
        # Info de vehículo
        'vehicle_zone': vehicle_zone,
        'zone_description': VEHICLE_ZONES.get(vehicle_zone, {}).get('description', 'unknown area'),
        'zone_area': VEHICLE_ZONES.get(vehicle_zone, {}).get('area', 'unknown'),
        
        # Defectos
        'total_defects': len(defects),
        'defect_types': defect_types,
        'defect_distribution': defect_distribution,
        'defects': defects,
        
        # Descripción textual
        'description': description,
        
        # Distribución espacial
        'spatial_distribution': spatial_dist,
        
        # Bboxes
        'bboxes': [d['bbox'] for d in defects],
        'total_bbox_area': sum([d['bbox_area'] for d in defects]),
        
        # Estadísticas
        'avg_defect_size': np.mean([d['bbox_area'] for d in defects]),
        'max_defect_size': max([d['bbox_area'] for d in defects]),
        'min_defect_size': min([d['bbox_area'] for d in defects])
    }
    
    return metadata


def process_no_damage_image(image_path: Path, json_path: Path) -> Dict:
    """Procesa imagen SIN daño (sin segmentaciones)"""
    
    # Leer JSON para obtener dimensiones (si existen)
    img_w, img_h = 0, 0
    if json_path.exists():
        try:
            with open(json_path) as f:
                json_data = json.load(f)
            img_w = json_data.get('imageWidth', 0)
            img_h = json_data.get('imageHeight', 0)
        except:
            pass
    
    # Si no hay dimensiones en JSON, leer de imagen
    if img_w == 0 or img_h == 0:
        from PIL import Image
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    
    vehicle_zone = extract_vehicle_zone(image_path.name)
    zone_info = VEHICLE_ZONES.get(vehicle_zone, {})
    
    metadata = {
        'image_path': str(image_path),
        'image_name': image_path.name,
        'image_size': [img_w, img_h],
        'has_damage': False,  # ← FLAG CRÍTICO
        
        # Info de vehículo
        'vehicle_zone': vehicle_zone,
        'zone_description': zone_info.get('description', 'unknown area'),
        'zone_area': zone_info.get('area', 'unknown'),
        
        # Sin defectos
        'total_defects': 0,
        'defect_types': [],
        'defect_distribution': {},
        'defects': [],
        
        # Descripción textual
        'description': f"No visible damage on {zone_info.get('description', 'vehicle')} ({zone_info.get('area', 'unknown')} area)",
        
        # Sin distribución espacial
        'spatial_distribution': {},
        'bboxes': [],
        'total_bbox_area': 0,
        'avg_defect_size': 0,
        'max_defect_size': 0,
        'min_defect_size': 0
    }
    
    return metadata


def calculate_spatial_distribution(defects: List[Dict], img_w: int, img_h: int) -> Dict:
    """Calcula distribución espacial de defectos en grilla 3×3"""
    from collections import defaultdict
    
    spatial_counts = defaultdict(int)
    
    for defect in defects:
        bbox = defect['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        rel_x = center_x / img_w if img_w > 0 else 0.5
        rel_y = center_y / img_h if img_h > 0 else 0.5
        
        zone_x = "left" if rel_x < 0.33 else "center" if rel_x < 0.67 else "right"
        zone_y = "top" if rel_y < 0.33 else "middle" if rel_y < 0.67 else "bottom"
        
        zone_key = f"{zone_y}_{zone_x}"
        spatial_counts[zone_key] += 1
    
    return dict(spatial_counts)


def build_damage_description(defects: List[Dict], vehicle_zone: str) -> str:
    """Construye descripción textual de defectos"""
    from collections import Counter
    
    type_counts = Counter([d['damage_type'] for d in defects])
    
    parts = []
    for dtype, count in type_counts.most_common():
        plural = "s" if count > 1 else ""
        parts.append(f"{count} {dtype}{plural}")
    
    damage_desc = " and ".join(parts) if parts else "no defects"
    
    zone_info = VEHICLE_ZONES.get(vehicle_zone, {})
    zone_desc = zone_info.get('description', 'unknown area')
    zone_area = zone_info.get('area', 'unknown')
    
    return f"{damage_desc} on {zone_desc} ({zone_area} area)"


def scan_dataset(dataset_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Escanea dataset y separa imágenes con/sin daño
    
    Returns:
        (damage_images, no_damage_images)
    """
    all_images = list(dataset_dir.glob("*_imageDANO_original.jpg"))
    
    damage_images = []
    no_damage_images = []
    
    for img_path in all_images:
        # Buscar JSON de daño
        json_damage = img_path.parent / img_path.name.replace(
            '_imageDANO_original.jpg',
            '_labelDANO_modificado.json'
        )
        
        # Buscar JSON sin daño
        json_no_damage = img_path.parent / img_path.name.replace(
            '_imageDANO_original.jpg',
            '_imageDANO_original.json'
        )
        
        if json_damage.exists():
            damage_images.append((img_path, json_damage))
        elif json_no_damage.exists():
            no_damage_images.append((img_path, json_no_damage))
        else:
            print(f"⚠️  Sin JSON asociado: {img_path.name}")
    
    return damage_images, no_damage_images


def generate_hybrid_embeddings_unified(
    dataset_dir: Path,
    output_dir: Path,
    visual_weight: float = 0.6,
    text_weight: float = 0.4,
    batch_size: int = 8
):
    """Pipeline completo de generación de embeddings híbridos (con/sin daño)"""
    
    print(f"\n{'='*70}")
    print(f"🌟 GENERACIÓN DE EMBEDDINGS HÍBRIDOS - DAÑO + NO DAÑO")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # 1. Escanear dataset
    print("📂 Escaneando dataset...")
    damage_images, no_damage_images = scan_dataset(dataset_dir)
    
    print(f"   ✅ Imágenes CON daño: {len(damage_images)}")
    print(f"   ✅ Imágenes SIN daño: {len(no_damage_images)}")
    print(f"   📊 Total: {len(damage_images) + len(no_damage_images)}\n")
    
    # 2. Procesar metadata
    print("📊 Procesando metadata...")
    
    all_metadata = []
    all_image_paths = []
    
    # Procesar imágenes CON daño
    print(f"   🔧 Procesando {len(damage_images)} imágenes con daño...")
    for img_path, json_path in damage_images:
        try:
            meta = process_damage_image(img_path, json_path)
            all_metadata.append(meta)
            all_image_paths.append(img_path)
        except Exception as e:
            print(f"   ⚠️  Error en {img_path.name}: {e}")
    
    # Procesar imágenes SIN daño
    print(f"   🔧 Procesando {len(no_damage_images)} imágenes sin daño...")
    for img_path, json_path in no_damage_images:
        try:
            meta = process_no_damage_image(img_path, json_path)
            all_metadata.append(meta)
            all_image_paths.append(img_path)
        except Exception as e:
            print(f"   ⚠️  Error en {img_path.name}: {e}")
    
    print(f"\n   ✅ Total metadata procesada: {len(all_metadata)}\n")
    
    # 3. Inicializar embedder
    print("🤖 Inicializando MultimodalEmbedderV2...")
    embedder = MultimodalEmbedderV2(
        visual_weight=visual_weight,
        text_weight=text_weight,
        use_bfloat16=True
    )
    print()
    
    # 4. Generar embeddings híbridos
    print(f"🧠 Generando embeddings híbridos (batch_size={batch_size})...")
    print(f"   - Visual weight: {visual_weight}")
    print(f"   - Text weight: {text_weight}")
    print(f"   - Total dim: {embedder.total_dim}\n")
    
    embeddings, debug_info = embedder.generate_batch_embeddings(
        image_paths=all_image_paths,
        metadata_list=all_metadata,
        batch_size=batch_size,
        show_progress=True
    )
    
    print(f"\n{'='*70}")
    print(f"✅ EMBEDDINGS HÍBRIDOS GENERADOS")
    print(f"{'='*70}")
    print(f"Shape: {embeddings.shape}")
    print(f"Dimensión: {embeddings.shape[1]}")
    print(f"  - Visual: {embedder.visual_dim} dims")
    print(f"  - Text: {embedder.text_dim} dims")
    print(f"{'='*70}\n")
    
    # 5. Estadísticas
    print("📊 Estadísticas de embeddings:")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"   - Norma promedio: {norms.mean():.4f}")
    print(f"   - Norma std: {norms.std():.4f}")
    print(f"   - Min norm: {norms.min():.4f}")
    print(f"   - Max norm: {norms.max():.4f}\n")
    
    # Estadísticas por tipo
    damage_indices = [i for i, m in enumerate(all_metadata) if m['has_damage']]
    no_damage_indices = [i for i, m in enumerate(all_metadata) if not m['has_damage']]
    
    print(f"📊 Estadísticas por tipo:")
    print(f"   CON daño ({len(damage_indices)} imgs):")
    if damage_indices:
        damage_norms = norms[damage_indices]
        print(f"     - Norma media: {damage_norms.mean():.4f}")
    
    print(f"   SIN daño ({len(no_damage_indices)} imgs):")
    if no_damage_indices:
        no_damage_norms = norms[no_damage_indices]
        print(f"     - Norma media: {no_damage_norms.mean():.4f}")
    print()
    
    # 6. Ejemplos de descripciones
    print("📝 Ejemplos de descripciones textuales:")
    print("-" * 70)
    
    # Ejemplo con daño
    damage_examples = [i for i, m in enumerate(all_metadata) if m['has_damage']][:2]
    for i in damage_examples:
        if i < len(debug_info) and 'text_description' in debug_info[i]:
            print(f"  [CON DAÑO] {debug_info[i]['text_description']}")
    
    # Ejemplo sin daño
    no_damage_examples = [i for i, m in enumerate(all_metadata) if not m['has_damage']][:2]
    for i in no_damage_examples:
        if i < len(debug_info) and 'text_description' in debug_info[i]:
            print(f"  [SIN DAÑO] {debug_info[i]['text_description']}")
    print()
    
    # 7. Enriquecer metadata
    for i, meta in enumerate(all_metadata):
        meta['embedding_index'] = i
        meta['embedding_model'] = 'multimodal_hybrid_v2'
        meta['embedding_dim'] = int(embeddings.shape[1])
        meta['embedding_norm'] = float(norms[i])
        meta['visual_weight'] = visual_weight
        meta['text_weight'] = text_weight
        
        if i < len(debug_info) and 'text_description' in debug_info[i]:
            meta['text_description_used'] = debug_info[i]['text_description']
    
    # 8. Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("💾 Guardando archivos...")
    
    # Embeddings
    embeddings_path = output_dir / "embeddings_hybrid_dano_nodano.npy"
    np.save(embeddings_path, embeddings)
    print(f"   ✅ Embeddings: {embeddings_path}")
    
    # Metadata
    metadata_path = output_dir / "metadata_hybrid_dano_nodano.json"
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"   ✅ Metadata: {metadata_path}")
    
    # Debug info
    debug_path = output_dir / "debug_info_dano_nodano.json"
    with open(debug_path, 'w') as f:
        json.dump(debug_info, f, indent=2)
    print(f"   ✅ Debug info: {debug_path}\n")
    
    # 9. Info del proceso
    process_info = {
        'timestamp': datetime.now().isoformat(),
        'model': embedder.get_model_info(),
        'dataset': {
            'dataset_dir': str(dataset_dir),
            'total_images': len(all_metadata),
            'damage_images': len(damage_indices),
            'no_damage_images': len(no_damage_indices)
        },
        'embeddings': {
            'shape': list(embeddings.shape),
            'dtype': str(embeddings.dtype),
            'norm_mean': float(norms.mean()),
            'norm_std': float(norms.std())
        },
        'weights': {
            'visual': visual_weight,
            'text': text_weight
        },
        'output_files': {
            'embeddings': str(embeddings_path),
            'metadata': str(metadata_path),
            'debug_info': str(debug_path)
        }
    }
    
    info_path = output_dir / "generation_info_dano_nodano.json"
    with open(info_path, 'w') as f:
        json.dump(process_info, f, indent=2)
    print(f"📋 Info del proceso: {info_path}\n")
    
    print(f"{'='*70}")
    print(f"✨ PROCESO COMPLETADO")
    print(f"{'='*70}\n")
    
    return embeddings, all_metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Genera embeddings híbridos para imágenes con/sin daño'
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        required=True,
        help='Directorio con dataset unificado (con/sin daño)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/processed/embeddings/hybrid_dano_nodano"),
        help='Directorio de salida'
    )
    parser.add_argument(
        '--visual-weight',
        type=float,
        default=0.6,
        help='Peso para embedding visual'
    )
    parser.add_argument(
        '--text-weight',
        type=float,
        default=0.4,
        help='Peso para embedding textual'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8
    )
    
    args = parser.parse_args()
    
    # Validar pesos
    if abs(args.visual_weight + args.text_weight - 1.0) > 1e-6:
        print(f"❌ Error: visual_weight + text_weight debe ser 1.0")
        print(f"   Actual: {args.visual_weight} + {args.text_weight} = {args.visual_weight + args.text_weight}")
        exit(1)
    
    try:
        generate_hybrid_embeddings_unified(
            dataset_dir=args.dataset_dir,
            output_dir=args.output,
            visual_weight=args.visual_weight,
            text_weight=args.text_weight,
            batch_size=args.batch_size
        )
        
        print("📌 Próximo paso:")
        print("   python scripts/04_build_faiss_index.py \\")
        print(f"       --embeddings {args.output}/embeddings_hybrid_dano_nodano.npy \\")
        print(f"       --metadata {args.output}/metadata_hybrid_dano_nodano.json \\")
        print(f"       --output outputs/vector_indices/hybrid_dano_nodano")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
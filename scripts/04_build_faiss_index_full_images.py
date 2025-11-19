#!/usr/bin/env python3
# scripts/04b_build_faiss_index_dano_nodano.py

"""
🏗️ CONSTRUCCIÓN ÍNDICE FAISS - DAÑO + NO DAÑO

Mejoras vs 04_build_faiss_index_full_images.py:
- Estadísticas diferenciadas para imágenes con/sin daño
- Validación de flag 'has_damage'
- Métricas más informativas
"""

from pathlib import Path
import json
import numpy as np
import faiss
import pickle
from typing import Dict

def build_faiss_index_unified(
    embeddings_path: Path,
    metadata_path: Path,
    output_dir: Path,
    index_type: str = "IndexHNSWFlat"
):
    """
    Construye índice FAISS con metadata enriquecida (daño + no daño)
    """
    
    print(f"\n{'='*70}")
    print(f"🏗️  CONSTRUCCIÓN ÍNDICE FAISS - DAÑO + NO DAÑO")
    print(f"{'='*70}\n")
    
    # 1. Cargar embeddings
    print("📊 Cargando embeddings...")
    embeddings = np.load(embeddings_path).astype('float32')
    n_vectors, dim = embeddings.shape
    print(f"   - Shape: {embeddings.shape}")
    print(f"   - Dtype: {embeddings.dtype}")
    
    # 2. Cargar metadata
    print("\n📄 Cargando metadata...")
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"   - Entries: {len(metadata)}")
    print(f"   - Tipo: Full Images + Metadata Enriquecida (CON/SIN daño)")
    
    # Verificar consistencia
    assert len(metadata) == n_vectors, \
        f"Mismatch: {len(metadata)} metadata vs {n_vectors} embeddings"
    
    # ✅ 3. Estadísticas del dataset (MEJORADAS)
    print("\n📊 Estadísticas del dataset:")
    
    # Separar por tipo
    damage_images = [m for m in metadata if m.get('has_damage', True)]
    no_damage_images = [m for m in metadata if not m.get('has_damage', True)]
    
    print(f"   - Total imágenes: {len(metadata)}")
    print(f"   - Imágenes CON daño: {len(damage_images)} ({len(damage_images)/len(metadata)*100:.1f}%)")
    print(f"   - Imágenes SIN daño: {len(no_damage_images)} ({len(no_damage_images)/len(metadata)*100:.1f}%)")
    
    # Estadísticas de defectos (solo para imágenes con daño)
    if damage_images:
        total_defects = sum(m['total_defects'] for m in damage_images)
        avg_defects = total_defects / len(damage_images)
        
        print(f"\n   📦 Estadísticas de defectos (solo imágenes CON daño):")
        print(f"   - Total defectos: {total_defects}")
        print(f"   - Promedio defectos/imagen: {avg_defects:.1f}")
        
        # Contar tipos únicos
        from collections import Counter
        all_types = []
        all_zones = []
        
        for m in damage_images:
            all_types.extend(m.get('defect_types', []))
            all_zones.append(m.get('vehicle_zone', 'unknown'))
        
        type_dist = Counter(all_types)
        zone_dist = Counter(all_zones)
        
        print(f"   - Tipos únicos detectados: {len(type_dist)}")
        print(f"   - Zonas vehículo cubiertas: {len(zone_dist)}")
        
        print(f"\n   Top-5 tipos de daño:")
        for dtype, count in type_dist.most_common(5):
            print(f"     - {dtype}: {count} ocurrencias")
    
    # ✅ Estadísticas de zonas (TODAS las imágenes)
    print(f"\n   📍 Distribución de zonas (TODAS las imágenes):")
    all_zones_unified = []
    for m in metadata:
        all_zones_unified.append(m.get('vehicle_zone', 'unknown'))
    
    zone_dist_unified = Counter(all_zones_unified)
    
    for zone, count in sorted(zone_dist_unified.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
        # Obtener descripción
        zone_desc = "unknown"
        for m in metadata:
            if m.get('vehicle_zone') == zone:
                zone_desc = m.get('zone_description', 'unknown')
                break
        
        # Calcular cuántas con/sin daño
        zone_damage = sum(1 for m in damage_images if m.get('vehicle_zone') == zone)
        zone_no_damage = sum(1 for m in no_damage_images if m.get('vehicle_zone') == zone)
        
        print(f"     - Zona {zone} ({zone_desc}): {count} imágenes "
              f"(CON daño: {zone_damage}, SIN daño: {zone_no_damage})")
    
    # 4. Construir índice
    print(f"\n🔧 Construyendo índice: {index_type}...")
    
    if index_type == "IndexHNSWFlat":
        # Óptimo para <10K vectores
        index = faiss.IndexHNSWFlat(dim, 32)  # M=32
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        
        print(f"   - Tipo: HNSW (Hierarchical Navigable Small World)")
        print(f"   - M: 32")
        print(f"   - efConstruction: 200")
        print(f"   - efSearch: 64")
        
    elif index_type == "IndexFlatL2":
        index = faiss.IndexFlatL2(dim)
        print(f"   - Tipo: Flat L2 (exhaustive search)")
        
    else:
        raise ValueError(f"Tipo no soportado: {index_type}")
    
    # 5. Añadir vectores
    print(f"\n➕ Añadiendo {n_vectors} vectores...")
    index.add(embeddings)
    print(f"   ✅ Total vectores: {index.ntotal}")
    
    # 6. Validación con búsqueda de prueba
    print("\n🔍 Validando índice...")
    test_query = embeddings[0:1]
    k = min(5, n_vectors)
    
    distances, indices = index.search(test_query, k)
    
    print(f"   - Query: embedding[0] ({'CON' if metadata[0].get('has_damage', True) else 'SIN'} daño)")
    print(f"   - Top-{k} resultados:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        meta = metadata[idx]
        has_dmg = "CON" if meta.get('has_damage', True) else "SIN"
        defects = meta.get('total_defects', 0)
        print(f"     {i+1}. Idx={idx}, Dist={dist:.4f}, {has_dmg} daño ({defects} defectos), Zone={meta.get('vehicle_zone', '?')}")
    
    # Validar que el primero es el mismo
    assert indices[0][0] == 0, "El primer resultado debería ser él mismo"
    assert distances[0][0] < 0.01, "La distancia a sí mismo debería ser ~0"
    print("   ✅ Validación exitosa")
    
    # 7. Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Índice FAISS
    index_filename = f"{index_type.lower()}_dano_nodano.index"
    index_path = output_dir / index_filename
    faiss.write_index(index, str(index_path))
    print(f"\n💾 Índice guardado: {index_path}")
    
    # Metadata (pickle)
    metadata_pkl_path = output_dir / "metadata_dano_nodano.pkl"
    with open(metadata_pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"📦 Metadata (pickle): {metadata_pkl_path}")
    
    # ✅ Config (MEJORADA)
    config = {
        "index_type": index_type,
        "data_type": "full_images_with_and_without_damage",
        "n_vectors": int(n_vectors),
        "embedding_dim": int(dim),
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
        
        # ✅ NUEVO: Estadísticas diferenciadas
        "dataset_stats": {
            "total_images": len(metadata),
            "damage_images": len(damage_images),
            "no_damage_images": len(no_damage_images),
            "damage_percentage": len(damage_images) / len(metadata) * 100 if metadata else 0,
        }
    }
    
    if damage_images:
        total_defects = sum(m['total_defects'] for m in damage_images)
        config["dataset_stats"]["total_defects"] = int(total_defects)
        config["dataset_stats"]["avg_defects_per_damage_image"] = float(total_defects / len(damage_images))
        
        # Distribución de tipos
        all_types = []
        for m in damage_images:
            all_types.extend(m.get('defect_types', []))
        
        type_dist = Counter(all_types)
        config["dataset_stats"]["damage_type_distribution"] = dict(type_dist)
    
    # Distribución de zonas (todas)
    config["dataset_stats"]["zone_distribution"] = dict(zone_dist_unified)
    
    if index_type == "IndexHNSWFlat":
        config.update({
            "M": 32,
            "efConstruction": 200,
            "efSearch": 64
        })
    
    config_path = output_dir / "index_config_dano_nodano.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"⚙️  Config: {config_path}")
    
    # 8. Stats finales
    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    
    print(f"\n{'='*70}")
    print(f"✅ ÍNDICE FAISS CONSTRUIDO")
    print(f"{'='*70}")
    print(f"📊 Resumen:")
    print(f"   - Tipo: Full Images (CON/SIN daño)")
    print(f"   - Vectores totales: {index.ntotal}")
    print(f"     • CON daño: {len(damage_images)} ({len(damage_images)/len(metadata)*100:.1f}%)")
    print(f"     • SIN daño: {len(no_damage_images)} ({len(no_damage_images)/len(metadata)*100:.1f}%)")
    print(f"   - Dimensión: {dim}")
    print(f"   - Tamaño: {index_size_mb:.2f} MB")
    if damage_images:
        print(f"   - Total defectos indexados: {sum(m['total_defects'] for m in damage_images)}")
    print(f"   - Metadata enriquecida: ✅")
    print(f"{'='*70}\n")
    
    return index, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embeddings',
        type=Path,
        default=Path("data/processed/embeddings/hybrid_dano_nodano/embeddings_hybrid_dano_nodano.npy"),
        help='Ruta a embeddings'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path("data/processed/embeddings/hybrid_dano_nodano/metadata_hybrid_dano_nodano.json"),
        help='Ruta a metadata'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("outputs/vector_indices/hybrid_dano_nodano"),
        help='Directorio de salida'
    )
    parser.add_argument(
        '--index-type',
        type=str,
        default="IndexHNSWFlat",
        choices=["IndexHNSWFlat", "IndexFlatL2"]
    )
    
    args = parser.parse_args()
    
    # Verificar archivos
    if not args.embeddings.exists():
        print(f"❌ Error: {args.embeddings} no existe")
        exit(1)
    
    if not args.metadata.exists():
        print(f"❌ Error: {args.metadata} no existe")
        exit(1)
    
    # Construir índice
    try:
        build_faiss_index_unified(
            embeddings_path=args.embeddings,
            metadata_path=args.metadata,
            output_dir=args.output,
            index_type=args.index_type
        )
        
        print("📌 Próximo paso:")
        print("   python scripts/05_evaluate_rag_hybrid.py \\")
        print(f"       --index {args.output}/indexhnswflat_dano_nodano.index \\")
        print(f"       --metadata {args.output}/metadata_dano_nodano.pkl")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
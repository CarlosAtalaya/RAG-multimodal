#!/usr/bin/env python3
# scripts/04_build_faiss_index.py (ACTUALIZADO para Full Images)

"""
🏗️ CONSTRUCCIÓN ÍNDICE FAISS - FULL IMAGES + METADATA ENRIQUECIDA

Mejoras vs versión anterior:
- Metadata con descripciones textuales
- Info de zonas del vehículo
- Distribución espacial de defectos
- Validación mejorada
"""

from pathlib import Path
import json
import numpy as np
import faiss
import pickle
from typing import Dict

def build_faiss_index(
    embeddings_path: Path,
    metadata_path: Path,
    output_dir: Path,
    index_type: str = "IndexHNSWFlat"
):
    """
    Construye índice FAISS con metadata enriquecida
    """
    
    print(f"\n{'='*70}")
    print(f"🏗️  CONSTRUCCIÓN ÍNDICE FAISS - FULL IMAGES")
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
    print(f"   - Tipo: Full Images + Metadata Enriquecida")
    
    # Verificar consistencia
    assert len(metadata) == n_vectors, \
        f"Mismatch: {len(metadata)} metadata vs {n_vectors} embeddings"
    
    # 3. Estadísticas del dataset
    print("\n📊 Estadísticas del dataset:")
    
    total_defects = sum(m['total_defects'] for m in metadata)
    avg_defects = total_defects / len(metadata)
    
    # Contar tipos únicos
    from collections import Counter
    all_types = []
    all_zones = []
    for m in metadata:
        all_types.extend(m['defect_types'])
        all_zones.append(m['vehicle_zone'])
    
    type_dist = Counter(all_types)
    zone_dist = Counter(all_zones)
    
    print(f"   - Total defectos: {total_defects}")
    print(f"   - Promedio defectos/imagen: {avg_defects:.1f}")
    print(f"   - Tipos únicos detectados: {len(type_dist)}")
    print(f"   - Zonas vehículo cubiertas: {len(zone_dist)}")
    
    print(f"\n   Top-5 tipos de daño:")
    for dtype, count in type_dist.most_common(5):
        print(f"     - {dtype}: {count} ocurrencias")
    
    print(f"\n   Distribución de zonas:")
    for zone, count in sorted(zone_dist.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
        zone_name = metadata[0].get('zone_description', 'unknown')  # Obtener descripción
        # Buscar descripción correcta
        for m in metadata:
            if m['vehicle_zone'] == zone:
                zone_name = m['zone_description']
                break
        print(f"     - Zona {zone} ({zone_name}): {count} imágenes")
    
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
    
    print(f"   - Query: embedding[0]")
    print(f"   - Top-{k} resultados:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        meta = metadata[idx]
        print(f"     {i+1}. Idx={idx}, Dist={dist:.4f}, Zone={meta['vehicle_zone']}, Defects={meta['total_defects']}")
    
    # Validar que el primero es el mismo
    assert indices[0][0] == 0, "El primer resultado debería ser él mismo"
    assert distances[0][0] < 0.01, "La distancia a sí mismo debería ser ~0"
    print("   ✅ Validación exitosa")
    
    # 7. Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Índice FAISS
    index_filename = f"{index_type.lower()}_fullimages.index"
    index_path = output_dir / index_filename
    faiss.write_index(index, str(index_path))
    print(f"\n💾 Índice guardado: {index_path}")
    
    # Metadata (pickle)
    metadata_pkl_path = output_dir / "metadata_fullimages.pkl"
    with open(metadata_pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"📦 Metadata (pickle): {metadata_pkl_path}")
    
    # Config
    config = {
        "index_type": index_type,
        "data_type": "full_images",
        "n_vectors": int(n_vectors),
        "embedding_dim": int(dim),
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
        "total_defects": int(total_defects),
        "avg_defects_per_image": float(avg_defects),
        "damage_type_distribution": dict(type_dist),
        "zone_distribution": dict(zone_dist)
    }
    
    if index_type == "IndexHNSWFlat":
        config.update({
            "M": 32,
            "efConstruction": 200,
            "efSearch": 64
        })
    
    config_path = output_dir / "index_config_fullimages.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"⚙️  Config: {config_path}")
    
    # 8. Stats
    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    
    print(f"\n{'='*70}")
    print(f"✅ ÍNDICE FAISS CONSTRUIDO")
    print(f"{'='*70}")
    print(f"📊 Resumen:")
    print(f"   - Tipo: Full Images (no crops)")
    print(f"   - Vectores: {index.ntotal}")
    print(f"   - Dimensión: {dim}")
    print(f"   - Tamaño: {index_size_mb:.2f} MB")
    print(f"   - Total defectos indexados: {total_defects}")
    print(f"   - Metadata enriquecida: ✅")
    print(f"{'='*70}\n")
    
    return index, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embeddings',
        type=Path,
        default=Path("data/processed/embeddings/fullimages_dinov3/embeddings_fullimages_dinov3.npy"),
        help='Ruta a embeddings'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path("data/processed/embeddings/fullimages_dinov3/metadata_fullimages.json"),
        help='Ruta a metadata'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("outputs/vector_indices/fullimages_dinov3"),
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
        build_faiss_index(
            embeddings_path=args.embeddings,
            metadata_path=args.metadata,
            output_dir=args.output,
            index_type=args.index_type
        )
        
        print("📌 Próximo paso:")
        print("   python scripts/06_evaluate_rag_end_to_end.py \\")
        print(f"       --index {args.output}/indexhnswflat_fullimages.index \\")
        print(f"       --metadata {args.output}/metadata_fullimages.pkl")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
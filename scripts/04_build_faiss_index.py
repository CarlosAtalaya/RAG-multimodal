# scripts/04_build_faiss_index.py

from pathlib import Path
import json
import numpy as np
import faiss
import pickle

def build_faiss_index(
    embeddings_path: Path,
    metadata_path: Path,
    output_dir: Path,
    index_type: str = "IndexHNSWFlat"
):
    """
    Construye índice FAISS para búsqueda de similitud
    
    Args:
        embeddings_path: Ruta a embeddings.npy
        metadata_path: Ruta a metadata JSON
        output_dir: Directorio de salida
        index_type: Tipo de índice FAISS
    """
    
    print(f"\n{'='*70}")
    print(f"🏗️  CONSTRUCCIÓN ÍNDICE FAISS")
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
    
    # Verificar consistencia
    assert len(metadata) == n_vectors, \
        f"Mismatch: {len(metadata)} metadata vs {n_vectors} embeddings"
    
    # 3. Construir índice según configuración
    print(f"\n🔧 Construyendo índice: {index_type}...")
    
    if index_type == "IndexHNSWFlat":
        # Óptimo para <10K vectores (POC)
        index = faiss.IndexHNSWFlat(dim, 32)  # M=32
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        
        print(f"   - Tipo: HNSW (Hierarchical Navigable Small World)")
        print(f"   - M: 32 (conectividad)")
        print(f"   - efConstruction: 200")
        print(f"   - efSearch: 64")
        
    elif index_type == "IndexFlatL2":
        # Búsqueda exhaustiva (baseline)
        index = faiss.IndexFlatL2(dim)
        print(f"   - Tipo: Flat L2 (exhaustive search)")
        
    else:
        raise ValueError(f"Tipo de índice no soportado: {index_type}")
    
    # 4. Añadir vectores al índice
    print(f"\n➕ Añadiendo {n_vectors} vectores al índice...")
    index.add(embeddings)
    print(f"   - Total vectores en índice: {index.ntotal}")
    
    # 5. Validar índice con búsqueda de prueba
    print("\n🔍 Validando índice con búsqueda de prueba...")
    test_query = embeddings[0:1]  # Primer embedding
    k = min(5, n_vectors)
    
    distances, indices = index.search(test_query, k)
    
    print(f"   - Query: embedding[0]")
    print(f"   - Top-{k} resultados:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"     {i+1}. Index={idx}, Distance={dist:.4f}")
    
    # El primer resultado debe ser él mismo (distancia ~0)
    assert indices[0][0] == 0, "El primer resultado debería ser el query mismo"
    assert distances[0][0] < 0.01, "La distancia a sí mismo debería ser ~0"
    print("   ✅ Validación exitosa")
    
    # 6. Guardar índice y metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar índice FAISS
    index_path = output_dir / f"{index_type.lower()}.index"
    faiss.write_index(index, str(index_path))
    print(f"\n💾 Índice guardado: {index_path}")
    
    # Guardar metadata asociada (pickle para preservar tipos)
    metadata_pkl_path = output_dir / "metadata.pkl"
    with open(metadata_pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"📦 Metadata (pickle): {metadata_pkl_path}")
    
    # Guardar configuración del índice
    config = {
        "index_type": index_type,
        "n_vectors": int(n_vectors),
        "embedding_dim": int(dim),
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path)
    }
    
    if index_type == "IndexHNSWFlat":
        config.update({
            "M": 32,
            "efConstruction": 200,
            "efSearch": 64
        })
    
    config_path = output_dir / "index_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"⚙️  Config: {config_path}")
    
    # 7. Estadísticas del índice
    print(f"\n{'='*70}")
    print(f"✅ ÍNDICE FAISS CONSTRUIDO EXITOSAMENTE")
    print(f"{'='*70}")
    print(f"📊 Estadísticas:")
    print(f"   - Vectores indexados: {index.ntotal}")
    print(f"   - Dimensión: {dim}")
    print(f"   - Tipo: {index_type}")
    
    # Tamaño del índice en disco
    index_size_mb = index_path.stat().st_size / (1024 * 1024)
    print(f"   - Tamaño en disco: {index_size_mb:.2f} MB")
    print(f"{'='*70}\n")
    
    return index, metadata

if __name__ == "__main__":
    # Rutas para mini-POC
    EMBEDDINGS_PATH = Path("data/processed/embeddings/embeddings_mini_100.npy")
    METADATA_PATH = Path("data/processed/embeddings/enriched_crops_metadata_mini_100.json")
    OUTPUT_DIR = Path("outputs/vector_indices")
    
    # Construir índice
    index, metadata = build_faiss_index(
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
        output_dir=OUTPUT_DIR,
        index_type="IndexHNSWFlat"  # Óptimo para POC
    )
    
    print("✨ Fase 4 completada!\n")
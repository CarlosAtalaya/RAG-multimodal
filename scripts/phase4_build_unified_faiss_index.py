# scripts/phase4_build_unified_faiss_index.py

import faiss
import numpy as np
import json
import pickle
from pathlib import Path

def main():
    # Rutas
    embeddings_path = Path("data/processed/embeddings/metaclip_unified/embeddings_metaclip_1024d.npy")
    metadata_path = Path("data/processed/embeddings/metaclip_unified/metadata_final_phase3.json")
    output_dir = Path("outputs/vector_indices/metaclip_unified_index")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not embeddings_path.exists():
        print("❌ No se encuentran los embeddings de Fase 3.")
        return

    print("🔹 Cargando embeddings y metadata...")
    embeddings = np.load(embeddings_path).astype('float32')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    d = embeddings.shape[1]
    count = embeddings.shape[0]
    print(f"   Dimensiones: {d}")
    print(f"   Total vectores: {count}")

    # Configuración Índice HNSW (Optimizado para retrieval rápido y preciso)
    # M=32 (conexiones), efConstruction=200 (precisión construcción)
    print("🔹 Construyendo IndexHNSWFlat...")
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200
    
    # Entrenar (no necesario para HNSWFlat pero buena práctica añadir en bloques si fuera muy grande)
    index.add(embeddings)
    
    # Configurar efSearch para tiempo de inferencia
    index.hnsw.efSearch = 64

    # Guardar
    index_file = output_dir / "index_metaclip_unified.faiss"
    metadata_pkl = output_dir / "metadata_unified.pkl"
    config_file = output_dir / "index_config.json"

    print(f"🔹 Guardando índice en {index_file}...")
    faiss.write_index(index, str(index_file))

    # Guardar metadata en pickle (más rápido de cargar en inferencia)
    with open(metadata_pkl, 'wb') as f:
        pickle.dump(metadata, f)

    # Guardar config para la API de inferencia
    config = {
        "index_type": "IndexHNSWFlat",
        "d": d,
        "metric": "InnerProduct" if "cosine" in str(index.metric_type) else "L2", # MetaCLIP normalizado funciona bien con IP (Cosine Sim) o L2
        "total_vectors": count,
        "model": "MetaCLIP 2 Unified"
    }
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n✅ Fase 4 Completada. Sistema RAG listo para consultas.")
    print(f"   Índice: {index_file}")

if __name__ == "__main__":
    main()
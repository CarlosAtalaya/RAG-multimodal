# scripts/03_generate_embeddings_mini.py

from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embeddings.hybrid_embedder import HybridEmbedder

def generate_mini_poc_embeddings(
    crops_metadata_path: Path,
    output_dir: Path,
    num_samples: int = 100  # 🔥 Solo 100 crops para validar
):
    """
    Genera embeddings para un subset pequeño (Mini-POC)
    """
    
    # Cargar metadata
    with open(crops_metadata_path) as f:
        crops_metadata = json.load(f)
    
    # 🔥 Tomar solo primeros N crops
    crops_metadata = crops_metadata[:num_samples]
    
    print(f"\n{'='*70}")
    print(f"🚀 MINI-POC: Generando embeddings para {len(crops_metadata)} crops")
    print(f"{'='*70}\n")
    
    # Inicializar embedder
    embedder = HybridEmbedder(
        qwen_api_endpoint="http://localhost:8001"
    )
    
    # Preparar datos
    image_paths = [Path(m['crop_path']) for m in crops_metadata]
    text_contexts = [
        f"{m['damage_type'].replace('_', ' ')} damage"
        for m in crops_metadata
    ]
    
    # Generar embeddings
    print("🧠 Generando embeddings...\n")
    embeddings = []
    
    for i in tqdm(range(len(image_paths)), desc="Embeddings"):
        try:
            embedding = embedder.generate_embedding(
                image_paths[i],
                text_contexts[i]
            )
            embeddings.append(embedding)
        except Exception as e:
            print(f"\n❌ Error en {image_paths[i].name}: {e}")
            embeddings.append(np.zeros(embedder.embedding_dim, dtype=np.float32))
    
    embeddings = np.vstack(embeddings)
    
    print(f"\n{'='*70}")
    print(f"✅ Generados {embeddings.shape[0]} embeddings de {embeddings.shape[1]} dims")
    print(f"{'='*70}\n")
    
    # Guardar embeddings
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / f"embeddings_mini_{num_samples}.npy"
    np.save(embeddings_path, embeddings)
    
    # Guardar metadata enriquecida
    for i, meta in enumerate(crops_metadata):
        meta['embedding_index'] = i
    
    enriched_metadata_path = output_dir / f"enriched_crops_metadata_mini_{num_samples}.json"
    with open(enriched_metadata_path, 'w') as f:
        json.dump(crops_metadata, f, indent=2)
    
    print(f"💾 Embeddings: {embeddings_path}")
    print(f"📄 Metadata: {enriched_metadata_path}\n")
    
    return embeddings, crops_metadata

if __name__ == "__main__":
    CROPS_METADATA = Path("data/processed/metadata/crops_metadata.json")
    OUTPUT_DIR = Path("data/processed/embeddings")
    
    # 🔥 Procesar solo 100 crops (~10-12 minutos)
    embeddings, metadata = generate_mini_poc_embeddings(
        CROPS_METADATA,
        OUTPUT_DIR,
        num_samples=100
    )
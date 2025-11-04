# scripts/03_generate_embeddings.py

from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embeddings.hybrid_embedder import HybridEmbedder

def generate_all_embeddings(
    crops_metadata_path: Path,
    output_dir: Path
):
    """
    Genera embeddings para todos los crops
    
    Soporta ambos tipos de metadata:
    - Individual: 'damage_type'
    - Clustering: 'dominant_type'
    """
    
    # Cargar metadata
    with open(crops_metadata_path) as f:
        crops_metadata = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"Generando embeddings para {len(crops_metadata)} crops")
    print(f"{'='*70}\n")
    
    # Inicializar embedder
    embedder = HybridEmbedder(
        qwen_api_endpoint="http://localhost:8001"
    )
    
    # Preparar datos
    image_paths = [Path(m['crop_path']) for m in crops_metadata]
    
    # Detectar tipo de metadata (individual vs clustering)
    is_clustering = 'dominant_type' in crops_metadata[0]
    
    if is_clustering:
        print("📦 Tipo de metadata: CLUSTERING")
        print(f"   - Clusters con múltiples defectos\n")
        text_contexts = [
            f"{m['dominant_type'].replace('_', ' ')} damage"
            for m in crops_metadata
        ]
    else:
        print("📦 Tipo de metadata: INDIVIDUAL")
        print(f"   - Un defecto por crop\n")
        text_contexts = [
            f"{m['damage_type'].replace('_', ' ')} damage"
            for m in crops_metadata
        ]
    
    # Generar embeddings
    print("🧠 Generando embeddings (esto puede tardar)...\n")
    embeddings = []
    failed_count = 0
    
    for i in tqdm(range(len(image_paths)), desc="Embeddings"):
        try:
            embedding = embedder.generate_embedding(
                image_paths[i],
                text_contexts[i]
            )
            embeddings.append(embedding)
        except Exception as e:
            print(f"\n❌ Error en {image_paths[i].name}: {e}")
            # Usar embedding zero si falla
            embeddings.append(np.zeros(embedder.embedding_dim, dtype=np.float32))
            failed_count += 1
    
    embeddings = np.vstack(embeddings)
    
    print(f"\n{'='*70}")
    print(f"✅ Generados {embeddings.shape[0]} embeddings de {embeddings.shape[1]} dims")
    if failed_count > 0:
        print(f"⚠️  Fallos: {failed_count} embeddings (usando zeros)")
    print(f"{'='*70}\n")
    
    # Guardar embeddings
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Nombre de archivo según tipo
    if is_clustering:
        embeddings_path = output_dir / "embeddings_clustered.npy"
        enriched_metadata_path = output_dir / "enriched_crops_metadata_clustered.json"
    else:
        embeddings_path = output_dir / "embeddings.npy"
        enriched_metadata_path = output_dir / "enriched_crops_metadata.json"
    
    np.save(embeddings_path, embeddings)
    
    # Guardar metadata enriquecida
    for i, meta in enumerate(crops_metadata):
        meta['embedding_index'] = i
    
    with open(enriched_metadata_path, 'w') as f:
        json.dump(crops_metadata, f, indent=2)
    
    print(f"💾 Embeddings: {embeddings_path}")
    print(f"📄 Metadata: {enriched_metadata_path}\n")
    
    return embeddings, crops_metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar embeddings de crops')
    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path("data/processed/metadata/clustered_crops_metadata.json"),
        help='Ruta a metadata de crops'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/processed/embeddings"),
        help='Directorio de salida'
    )
    
    args = parser.parse_args()
    
    # Verificar que existe el archivo
    if not args.metadata.exists():
        print(f"❌ Error: No se encontró {args.metadata}")
        print("\n💡 Opciones disponibles:")
        print("   - Individual: data/processed/metadata/crops_metadata.json")
        print("   - Clustering: data/processed/metadata/clustered_crops_metadata.json")
        sys.exit(1)
    
    embeddings, metadata = generate_all_embeddings(
        args.metadata,
        args.output
    )
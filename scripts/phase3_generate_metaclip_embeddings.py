# scripts/phase3_generate_metaclip_embeddings.py

import json
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from src.core.embeddings.metaclip_embedder_unified import MetaCLIPUnifiedEmbedder

def main():
    # Rutas
    input_metadata = Path("data/processed/metadata/metadata_crops_enriched_phase2.json")
    output_dir = Path("data/processed/embeddings/metaclip_unified")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_file = output_dir / "embeddings_metaclip_1024d.npy"
    final_metadata_file = output_dir / "metadata_final_phase3.json"

    if not input_metadata.exists():
        print("❌ Ejecuta primero la Fase 2 para generar contextos.")
        return

    # Cargar metadata
    with open(input_metadata, 'r') as f:
        metadata = json.load(f)

    # Inicializar Embedder
    embedder = MetaCLIPUnifiedEmbedder()
    
    embeddings_list = []
    metadata_updated = []

    print(f"🚀 Iniciando generación de embeddings MetaCLIP para {len(metadata)} crops...")

    for item in tqdm(metadata, desc="Embedding"):
        crop_path = Path(item['crop_path'])
        text_desc = item.get('text_description', '')

        if not crop_path.exists():
            print(f"⚠️ Aviso: Imagen no encontrada {crop_path}, saltando...")
            continue

        # Generar vector unificado (Average Fusion)
        vector = embedder.generate_embedding(str(crop_path), text_desc, fusion='average')
        
        embeddings_list.append(vector)
        
        # Guardar metadata con referencia
        item['embedding_dims'] = vector.shape[0]
        item['embedding_model'] = embedder.MODEL_NAME
        metadata_updated.append(item)

    # Convertir a numpy array
    embeddings_np = np.vstack(embeddings_list)

    # Guardar resultados
    np.save(embeddings_file, embeddings_np)
    with open(final_metadata_file, 'w') as f:
        json.dump(metadata_updated, f, indent=2)

    print(f"\n✅ Fase 3 Completada.")
    print(f"   Embeddings shape: {embeddings_np.shape}")
    print(f"   Guardado en: {output_dir}")

if __name__ == "__main__":
    main()
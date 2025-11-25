# scripts/phase2_generate_contexts.py

import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.embeddings.crop_contextualizer import CropContextualizer

def main():
    # Rutas
    input_metadata = Path("data/processed/crops/balanced_optimized_v2/metadata_crops_phase1.json")
    output_metadata = Path("data/processed/metadata/metadata_crops_enriched_phase2.json")
    
    if not input_metadata.exists():
        raise FileNotFoundError(f"No se encuentra la metadata de Fase 1: {input_metadata}")

    print(f"🔹 Cargando metadata Fase 1: {input_metadata}")
    with open(input_metadata, 'r') as f:
        metadata_list = json.load(f)

    # Inicializar contextualizador
    contextualizer = CropContextualizer()

    print(f"🔹 Generando contextos para {len(metadata_list)} crops...")
    
    enriched_count = 0
    for item in tqdm(metadata_list, desc="Contextualizando"):
        # Generar descripción
        text_desc = contextualizer.build_context(item)
        
        # Inyectar en metadata
        item['text_description'] = text_desc
        enriched_count += 1
        
        # Debug print de los primeros 2 items
        if enriched_count <= 2:
            print(f"\n[Ejemplo {enriched_count}] Tipo: {'DAÑO' if item['has_damage'] else 'LIMPIO'}")
            print(f"Prompt: {text_desc}")

    # Guardar
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metadata, 'w') as f:
        json.dump(metadata_list, f, indent=2)

    print(f"\n✅ Fase 2 Completada. Metadata enriquecida guardada en: {output_metadata}")

if __name__ == "__main__":
    main()
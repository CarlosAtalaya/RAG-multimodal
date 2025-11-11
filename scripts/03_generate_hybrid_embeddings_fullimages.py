#!/usr/bin/env python3
# scripts/03d_generate_hybrid_embeddings.py

"""
🌟 GENERACIÓN DE EMBEDDINGS HÍBRIDOS (Visual + Textual)

Pipeline:
1. Carga metadata de full images (con descripciones)
2. Para cada imagen:
   - Genera embedding visual (DINOv3)
   - Construye descripción textual rica
   - Genera embedding textual (Sentence-BERT)
   - Fusiona con pesos (0.6 visual + 0.4 text)
3. Guarda embeddings híbridos (1408 dims)

Esperado: Mejora de Recall@5 de 0% → 50-65%
"""

from pathlib import Path
import json
import numpy as np
import sys
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embeddings.multimodal_embedder import MultimodalEmbedder


def generate_hybrid_embeddings(
    train_dir: Path,
    output_dir: Path,
    visual_weight: float = 0.6,
    text_weight: float = 0.4,
    batch_size: int = 8
):
    """Pipeline completo de generación de embeddings híbridos"""
    
    print(f"\n{'='*70}")
    print(f"🌟 GENERACIÓN DE EMBEDDINGS HÍBRIDOS (Visual + Text)")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # 1. Cargar metadata existente
    print("📄 Cargando metadata de full images...")
    metadata_path = Path("data/processed/embeddings/fullimages_dinov3/metadata_fullimages.json")
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata no encontrada: {metadata_path}\n"
            "Ejecuta primero: python scripts/03c_generate_fullimage_embeddings.py"
        )
    
    with open(metadata_path) as f:
        metadata_list = json.load(f)
    
    print(f"   ✅ {len(metadata_list)} entradas cargadas\n")
    
    # 2. Verificar imágenes
    print("🔍 Verificando imágenes...")
    valid_data = []
    image_paths = []
    
    for meta in metadata_list:
        img_path = Path(meta['image_path'])
        if img_path.exists():
            valid_data.append(meta)
            image_paths.append(img_path)
        else:
            print(f"   ⚠️  Imagen no encontrada: {img_path.name}")
    
    print(f"   ✅ {len(valid_data)} imágenes válidas\n")
    
    # 3. Inicializar embedder multimodal
    embedder = MultimodalEmbedder(
        visual_weight=visual_weight,
        text_weight=text_weight,
        use_bfloat16=True
    )
    
    # 4. Generar embeddings híbridos
    print(f"🧠 Generando embeddings híbridos (batch_size={batch_size})...")
    print(f"   - Visual weight: {visual_weight}")
    print(f"   - Text weight: {text_weight}")
    print(f"   - Total dim: {embedder.total_dim}\n")
    
    embeddings, debug_info = embedder.generate_batch_embeddings(
        image_paths=image_paths,
        metadata_list=valid_data,
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
    
    # 6. Ejemplos de descripciones textuales
    print("📝 Ejemplos de descripciones textuales generadas:")
    print("-" * 70)
    for i in range(min(5, len(debug_info))):
        if 'text_description' in debug_info[i]:
            print(f"  {i+1}. {debug_info[i]['text_description']}")
    print()
    
    # 7. Enriquecer metadata con info híbrida
    for i, meta in enumerate(valid_data):
        meta['embedding_index'] = i
        meta['embedding_model'] = 'multimodal_hybrid'
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
    embeddings_path = output_dir / "embeddings_hybrid_dinov3_text.npy"
    np.save(embeddings_path, embeddings)
    print(f"   ✅ Embeddings: {embeddings_path}")
    
    # Metadata
    metadata_path = output_dir / "metadata_hybrid.json"
    with open(metadata_path, 'w') as f:
        json.dump(valid_data, f, indent=2)
    print(f"   ✅ Metadata: {metadata_path}")
    
    # Debug info
    debug_path = output_dir / "debug_info.json"
    with open(debug_path, 'w') as f:
        json.dump(debug_info, f, indent=2)
    print(f"   ✅ Debug info: {debug_path}\n")
    
    # 9. Info del proceso
    process_info = {
        'timestamp': datetime.now().isoformat(),
        'model': embedder.get_model_info(),
        'dataset': {
            'total_images': len(metadata_list),
            'valid_images': len(valid_data),
            'train_dir': str(train_dir)
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
    
    info_path = output_dir / "generation_info.json"
    with open(info_path, 'w') as f:
        json.dump(process_info, f, indent=2)
    print(f"📋 Info del proceso: {info_path}\n")
    
    print(f"{'='*70}")
    print(f"✨ PROCESO COMPLETADO")
    print(f"{'='*70}\n")
    
    return embeddings, valid_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Genera embeddings híbridos (visual + textual)'
    )
    parser.add_argument(
        '--train-dir',
        type=Path,
        default=Path("data/raw/train_test_split_8020/train"),
        help='Directorio con train set'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/processed/embeddings/hybrid_dinov3_text"),
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
        generate_hybrid_embeddings(
            train_dir=args.train_dir,
            output_dir=args.output,
            visual_weight=args.visual_weight,
            text_weight=args.text_weight,
            batch_size=args.batch_size
        )
        
        print("📌 Próximo paso:")
        print("   python scripts/04_build_faiss_index.py \\")
        print(f"       --embeddings {args.output}/embeddings_hybrid_dinov3_text.npy \\")
        print(f"       --metadata {args.output}/metadata_hybrid.json \\")
        print(f"       --output outputs/vector_indices/hybrid_dinov3_text")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
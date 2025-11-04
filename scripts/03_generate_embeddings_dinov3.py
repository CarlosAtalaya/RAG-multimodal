# scripts/03_generate_embeddings_dinov3.py

from pathlib import Path
import json
import numpy as np
import sys
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embeddings.dinov3_vitl_embedder import DINOv3ViTLEmbedder

def generate_embeddings_dinov3_vitl(
    crops_metadata_path: Path,
    output_dir: Path,
    batch_size: int = 16,
    dry_run: bool = False
):
    """
    Genera embeddings usando DINOv3-ViT-L/16
    
    Args:
        crops_metadata_path: Ruta a metadata de crops
        output_dir: Directorio de salida
        batch_size: Tamaño del batch (16 recomendado para ViT-L)
        dry_run: Si True, solo valida sin generar embeddings
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 GENERACIÓN DE EMBEDDINGS - DINOv3-ViT-L/16")
    print(f"{'='*70}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # 1. Cargar metadata
    print("📄 Cargando metadata...")
    with open(crops_metadata_path) as f:
        crops_metadata = json.load(f)
    
    total_crops = len(crops_metadata)
    print(f"   ✅ {total_crops} crops encontrados\n")
    
    # 2. Preparar rutas de imágenes
    print("📂 Validando rutas de imágenes...")
    image_paths = []
    missing_count = 0
    
    for meta in crops_metadata:
        path = Path(meta['crop_path'])
        if path.exists():
            image_paths.append(path)
        else:
            missing_count += 1
            if missing_count == 1:
                print(f"   ⚠️  Primera imagen faltante: {path}")
    
    if missing_count > 0:
        print(f"   ⚠️  {missing_count} imágenes no encontradas (de {total_crops})")
    else:
        print(f"   ✅ Todas las imágenes encontradas")
    
    valid_crops = len(image_paths)
    print(f"   📊 Crops válidos: {valid_crops}\n")
    
    if dry_run:
        print("🔍 Modo DRY RUN - No se generarán embeddings")
        print(f"   Batch size: {batch_size}")
        print(f"   Batches estimados: {(valid_crops + batch_size - 1) // batch_size}")
        return None, None
    
    # 3. Inicializar embedder
    print("🤖 Inicializando DINOv3-ViT-L/16...\n")
    start_init = time.time()
    
    embedder = DINOv3ViTLEmbedder(use_bfloat16=True)
    
    init_time = time.time() - start_init
    print(f"⏱️  Tiempo de inicialización: {init_time:.2f}s\n")
    
    # 4. Generar embeddings
    print(f"🧠 Generando embeddings (batch_size={batch_size})...\n")
    start_embed = time.time()
    
    embeddings = embedder.generate_batch_embeddings(
        image_paths,
        batch_size=batch_size,
        normalize=True,
        show_progress=True
    )
    
    embed_time = time.time() - start_embed
    
    print(f"\n{'='*70}")
    print(f"✅ EMBEDDINGS GENERADOS")
    print(f"{'='*70}")
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
    print(f"Dimensión: {embeddings.shape[1]}")
    print(f"⏱️  Tiempo total: {embed_time:.2f}s")
    print(f"⏱️  Tiempo/imagen: {embed_time/valid_crops:.3f}s")
    print(f"{'='*70}\n")
    
    # 5. Estadísticas de embeddings
    print("📊 Estadísticas de embeddings:")
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"   - Norma promedio: {norms.mean():.4f}")
    print(f"   - Norma std: {norms.std():.4f}")
    print(f"   - Norma min: {norms.min():.4f}")
    print(f"   - Norma max: {norms.max():.4f}")
    print(f"   - Valor min: {embeddings.min():.4f}")
    print(f"   - Valor max: {embeddings.max():.4f}\n")
    
    # 6. Guardar embeddings
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("💾 Guardando archivos...")
    
    # Embeddings
    embeddings_filename = "embeddings_dinov3_vitl.npy"
    embeddings_path = output_dir / embeddings_filename
    np.save(embeddings_path, embeddings)
    print(f"   ✅ Embeddings: {embeddings_path}")
    
    # Metadata enriquecida
    for i, meta in enumerate(crops_metadata):
        if i < len(embeddings):  # Solo para crops válidos
            meta['embedding_index'] = i
            meta['embedding_model'] = 'dinov3-vitl16'
            meta['embedding_model_full'] = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
            meta['embedding_dim'] = int(embeddings.shape[1])
            meta['embedding_norm'] = float(norms[i])
    
    metadata_filename = "enriched_crops_metadata_dinov3_vitl.json"
    enriched_path = output_dir / metadata_filename
    
    with open(enriched_path, 'w') as f:
        json.dump(crops_metadata, f, indent=2)
    print(f"   ✅ Metadata: {enriched_path}\n")
    
    # 7. Guardar info del proceso
    process_info = {
        'timestamp': datetime.now().isoformat(),
        'model': embedder.get_model_info(),
        'dataset': {
            'total_crops': total_crops,
            'valid_crops': valid_crops,
            'missing_crops': missing_count,
            'metadata_path': str(crops_metadata_path)
        },
        'processing': {
            'batch_size': batch_size,
            'total_time_seconds': embed_time,
            'time_per_image_seconds': embed_time / valid_crops,
            'init_time_seconds': init_time
        },
        'embeddings': {
            'shape': list(embeddings.shape),
            'dtype': str(embeddings.dtype),
            'norm_mean': float(norms.mean()),
            'norm_std': float(norms.std())
        },
        'output_files': {
            'embeddings': str(embeddings_path),
            'metadata': str(enriched_path)
        }
    }
    
    info_path = output_dir / "embedding_generation_info.json"
    with open(info_path, 'w') as f:
        json.dump(process_info, f, indent=2)
    print(f"📋 Info del proceso: {info_path}\n")
    
    print(f"{'='*70}")
    print(f"✨ PROCESO COMPLETADO EXITOSAMENTE")
    print(f"{'='*70}\n")
    
    return embeddings, crops_metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generar embeddings con DINOv3-ViT-L/16',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  
  # Uso básico
  python scripts/03_generate_embeddings_dinov3.py
  
  # Con paths personalizados
  python scripts/03_generate_embeddings_dinov3.py \\
      --metadata data/processed/metadata/crops_metadata.json \\
      --output data/processed/embeddings
  
  # Ajustar batch size según tu GPU
  python scripts/03_generate_embeddings_dinov3.py --batch-size 8
  
  # Modo dry-run (solo validación)
  python scripts/03_generate_embeddings_dinov3.py --dry-run
        """
    )
    
    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path("data/processed/crops/high_density_20samples/metadata/clustered_crops_metadata.json"),
        help='Ruta a metadata de crops (default: clustered_crops_metadata.json)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/processed/embeddings"),
        help='Directorio de salida (default: data/processed/embeddings)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Tamaño del batch (default: 16, recomendado para ViT-L)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Solo validar sin generar embeddings'
    )
    
    args = parser.parse_args()
    
    # Verificar que existe el archivo de metadata
    if not args.metadata.exists():
        print(f"❌ Error: No se encontró el archivo de metadata")
        print(f"   Ruta buscada: {args.metadata}")
        print(f"\n💡 Opciones disponibles:")
        
        metadata_dir = Path("data/processed/metadata")
        if metadata_dir.exists():
            json_files = list(metadata_dir.glob("*.json"))
            if json_files:
                print("   Archivos encontrados:")
                for f in json_files:
                    print(f"   - {f}")
        
        sys.exit(1)
    
    # Ejecutar generación
    try:
        embeddings, metadata = generate_embeddings_dinov3_vitl(
            crops_metadata_path=args.metadata,
            output_dir=args.output,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        if not args.dry_run:
            print("🎉 Éxito! Embeddings generados correctamente\n")
            print("📌 Próximos pasos:")
            print("   1. Construir índice FAISS:")
            print(f"      python scripts/04_build_faiss_index.py \\")
            print(f"          --embeddings {args.output}/embeddings_dinov3_vitl.npy \\")
            print(f"          --metadata {args.output}/enriched_crops_metadata_dinov3_vitl.json")
            print()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante la generación:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        print("\n📋 Traceback completo:")
        traceback.print_exc()
        sys.exit(1)
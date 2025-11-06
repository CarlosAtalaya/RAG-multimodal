# scripts/validate_taxonomy.py

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag.retriever import DamageRAGRetriever


def validate_index(index_path: Path, metadata_path: Path):
    """Valida cobertura taxonómica de un índice"""
    
    print("\n" + "="*70)
    print("🔍 VALIDACIÓN TAXONÓMICA")
    print("="*70 + "\n")
    
    # Cargar retriever con normalización
    retriever = DamageRAGRetriever(
        index_path=index_path,
        metadata_path=metadata_path,
        enable_taxonomy_normalization=True
    )
    
    # Stats
    stats = retriever.get_stats()
    
    print(f"\n📊 Estadísticas:")
    print(f"   Vectores: {stats['n_vectors']}")
    print(f"   Dimensión: {stats['embedding_dim']}")
    print(f"   Normalización: {'✅' if stats['normalization_enabled'] else '❌'}")
    
    if 'taxonomy_coverage' in stats:
        cov = stats['taxonomy_coverage']
        print(f"\n📈 Cobertura:")
        print(f"   Total: {cov['total_samples']}")
        print(f"   Mapeados: {cov['mapped_samples']} ({cov['coverage_percent']:.1f}%)")
        print(f"   Sin mapeo: {cov['unmapped_samples']}")
        
        if cov['unmapped_samples'] > 0:
            print(f"\n⚠️  Labels sin mapeo detectados")
        
        print(f"\n✅ Distribución:")
        for label, count in sorted(
            cov['label_distribution'].items(), 
            key=lambda x: -x[1]
        )[:10]:
            print(f"   {label:20} → {count:5}")
    
    print("\n" + "="*70)
    print("✅ Validación completada")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--index',
        type=Path,
        default=Path("outputs/vector_indices/train_set_dinov3/indexhnswflat_clustered.index")
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path("outputs/vector_indices/train_set_dinov3/metadata_clustered.pkl")
    )
    
    args = parser.parse_args()
    
    if not args.index.exists():
        print(f"❌ Índice no encontrado: {args.index}")
        sys.exit(1)
    
    if not args.metadata.exists():
        print(f"❌ Metadata no encontrada: {args.metadata}")
        sys.exit(1)
    
    validate_index(args.index, args.metadata)
# scripts/test_normalized_retrieval.py

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag.retriever import DamageRAGRetriever
from src.core.embeddings.dinov3_vitl_embedder import DINOv3ViTLEmbedder


def test_retrieval():
    """Test completo de retrieval con normalización"""
    
    print("\n" + "="*70)
    print("🧪 TEST: RETRIEVAL CON NORMALIZACIÓN TAXONÓMICA")
    print("="*70 + "\n")
    
    # Cargar retriever
    retriever = DamageRAGRetriever(
        index_path=Path("outputs/vector_indices/train_set_dinov3/indexhnswflat_clustered.index"),
        metadata_path=Path("outputs/vector_indices/train_set_dinov3/metadata_clustered.pkl"),
        enable_taxonomy_normalization=True
    )
    
    # Cargar embeddings de test
    test_embeddings = np.load(
        "data/processed/embeddings/dinov3_embeddings/train_815_originalimages/embeddings_dinov3_vitl.npy"
    )
    
    # Test con primer embedding
    query_emb = test_embeddings[0]
    
    print("\n🔍 Búsqueda Top-5:\n")
    results = retriever.search(query_emb, k=5, return_normalized=True)
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.damage_type:20} (original: {r.damage_type_original:20})")
        print(f"   Similitud: {(1 - r.distance)*100:.1f}%")
        print(f"   Confianza: {r.damage_type_confidence:.0%}")
        print(f"   Zona: {r.spatial_zone}")
        if r.all_damage_types:
            print(f"   Cluster: {r.all_damage_types}")
        print()
    
    # Construir contexto RAG
    print("="*70)
    print("📝 CONTEXTO RAG GENERADO")
    print("="*70)
    
    context = retriever.build_rag_context(results, max_examples=3)
    print(context)
    
    print("\n" + "="*70)
    print("✅ Test completado")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_retrieval()
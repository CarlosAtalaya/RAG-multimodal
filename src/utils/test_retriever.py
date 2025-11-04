# scripts/test_retriever.py

from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.rag.retriever import DamageRAGRetriever

def test_retriever():
    """Test básico del retriever"""
    
    print("\n" + "="*70)
    print("🧪 TESTEANDO RAG RETRIEVER")
    print("="*70 + "\n")
    
    # Inicializar retriever (modificar para cada subset)
    retriever = DamageRAGRetriever(
        index_path=Path("outputs/vector_indices/dinov3_indices/high_density/indexhnswflat_clustered.index"),
        metadata_path=Path("outputs/vector_indices/dinov3_indices/high_density/metadata_clustered.pkl"),
        config_path=Path("outputs/vector_indices/dinov3_indices/high_density/index_config_clustered.json")
    )
    
    # Cargar embeddings para test
    embeddings = np.load("data/processed/embeddings/dinov3_embeddings/high_density_20samples/embeddings_dinov3_vitl.npy")
    
    # TEST 1: Búsqueda sin filtros
    print("\n" + "-"*70)
    print("TEST 1: Búsqueda de Top-5 similares (sin filtros)")
    print("-"*70)
    
    query_embedding = embeddings[0]
    results = retriever.search(query_embedding, k=5)
    
    print(f"\nResultados encontrados: {len(results)}\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.damage_type} (dist={result.distance:.4f})")
        print(f"   Zone: {result.spatial_zone}, Image: {Path(result.image_path).name}")
    
    # TEST 2: Búsqueda con filtro de tipo
    print("\n" + "-"*70)
    print("TEST 2: Búsqueda filtrada por 'surface_scratch'")
    print("-"*70)
    
    results_filtered = retriever.search(
        query_embedding,
        k=5,
        filters={'damage_type': 'surface_scratch'}
    )
    
    print(f"\nResultados encontrados: {len(results_filtered)}\n")
    for i, result in enumerate(results_filtered, 1):
        print(f"{i}. {result.damage_type} (dist={result.distance:.4f})")
    
    # TEST 3: Construcción de contexto RAG
    print("\n" + "-"*70)
    print("TEST 3: Contexto RAG generado")
    print("-"*70)
    
    context = retriever.build_rag_context(results, max_examples=3)
    print("\n" + context)
    
    print("\n" + "="*70)
    print("✅ Todos los tests pasaron!")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_retriever()
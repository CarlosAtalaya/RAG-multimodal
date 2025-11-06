# diagnosis/test_embedding_alignment.py

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.embeddings.dinov3_vitl_embedder import DINOv3ViTLEmbedder
from scipy.spatial.distance import cosine

def test_crop_vs_fullimage_similarity():
    """
    Compara embeddings de:
    1. Imagen completa (como en evaluación)
    2. Crop del mismo defecto (como en índice FAISS)
    
    Hipótesis: Si la similitud es baja (<0.7), el RAG recupera mal.
    """
    
    embedder = DINOv3ViTLEmbedder()
    
    # Cargar metadata para encontrar crops y sus imágenes fuente
    import pickle
    metadata_path = Path("outputs/vector_indices/train_set_dinov3/metadata_clustered.pkl")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Seleccionar 20 crops aleatorios
    import random
    random.seed(42)
    sample_size = 20
    sampled_crops = random.sample(metadata, min(sample_size, len(metadata)))
    
    similarities = []
    
    print("="*70)
    print("ANÁLISIS DE SIMILITUD: CROP vs IMAGEN COMPLETA")
    print("="*70 + "\n")
    
    for i, crop_meta in enumerate(sampled_crops, 1):
        crop_path = Path(crop_meta['crop_path'])
        source_image = Path(crop_meta['source_image'])
        
        if not crop_path.exists() or not source_image.exists():
            continue
        
        # Generar embeddings
        emb_crop = embedder.generate_embedding(crop_path, normalize=True)
        emb_full = embedder.generate_embedding(source_image, normalize=True)
        
        # Calcular similitud coseno
        similarity = 1 - cosine(emb_crop, emb_full)
        similarities.append(similarity)
        
        print(f"[{i:2d}/{sample_size}] Crop: {crop_path.name[:40]:40} | Similarity: {similarity:.4f}")
    
    # Estadísticas
    similarities = np.array(similarities)
    
    print("\n" + "="*70)
    print("ESTADÍSTICAS")
    print("="*70)
    print(f"Mean similarity:   {similarities.mean():.4f}")
    print(f"Std deviation:     {similarities.std():.4f}")
    print(f"Min similarity:    {similarities.min():.4f}")
    print(f"Max similarity:    {similarities.max():.4f}")
    print(f"Median:            {np.median(similarities):.4f}")
    
    # Interpretación
    print("\n" + "="*70)
    print("INTERPRETACIÓN")
    print("="*70)
    
    if similarities.mean() < 0.6:
        print("❌ Similitud MUY BAJA (<0.6)")
        print("   → El RAG está recuperando crops con baja correlación semántica")
        print("   → RECOMENDACIÓN: Indexar imágenes completas en lugar de crops")
    elif similarities.mean() < 0.75:
        print("⚠️  Similitud MODERADA (0.6-0.75)")
        print("   → El RAG puede recuperar crops parcialmente relevantes")
        print("   → RECOMENDACIÓN: Considerar embeddings jerárquicos (global + local)")
    else:
        print("✅ Similitud ALTA (>0.75)")
        print("   → La representación de crops es suficientemente similar")
        print("   → El problema está probablemente en la taxonomía")
    
    return similarities

if __name__ == "__main__":
    similarities = test_crop_vs_fullimage_similarity()
#!/usr/bin/env python3
# diagnosis/test_fullimage_embedding_consistency.py

"""
🔬 VALIDACIÓN DE CONSISTENCIA: FULL IMAGES vs FULL IMAGES

Pruebas:
1. Self-similarity: Embedding de misma imagen debe ser ~1.0
2. Train-Train similarity: Imágenes del train set entre sí
3. Test-Train similarity: Query del test vs imágenes del train
4. Retrieval quality: Top-k resultados tienen sentido semántico

Objetivo: Validar que el sistema full images funciona correctamente
"""

import numpy as np
from pathlib import Path
import sys
import json
import pickle
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.embeddings.dinov3_vitl_embedder import DINOv3ViTLEmbedder
from scipy.spatial.distance import cosine
from collections import Counter


def test_self_similarity():
    """
    TEST 1: Self-Similarity
    
    Genera embedding de la misma imagen 2 veces.
    Esperado: similarity ≈ 1.0 (o muy cercano)
    """
    print("="*70)
    print("TEST 1: SELF-SIMILARITY (misma imagen, 2 embeddings)")
    print("="*70 + "\n")
    
    embedder = DINOv3ViTLEmbedder()
    
    # Cargar metadata del train
    metadata_path = Path("outputs/vector_indices/fullimages_dinov3/metadata_fullimages.pkl")
    
    if not metadata_path.exists():
        print(f"❌ Metadata no encontrada: {metadata_path}")
        return None
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Seleccionar 5 imágenes aleatorias
    import random
    random.seed(42)
    sample_size = 5
    sampled = random.sample(metadata, min(sample_size, len(metadata)))
    
    similarities = []
    
    for i, meta in enumerate(sampled, 1):
        image_path = Path(meta['image_path'])
        
        if not image_path.exists():
            print(f"   ⚠️  Imagen no encontrada: {image_path.name}")
            continue
        
        # Generar 2 embeddings de la misma imagen
        emb1 = embedder.generate_embedding(image_path, normalize=True)
        emb2 = embedder.generate_embedding(image_path, normalize=True)
        
        similarity = 1 - cosine(emb1, emb2)
        similarities.append(similarity)
        
        status = "✅" if similarity > 0.99 else "⚠️"
        print(f"{status} [{i}/{sample_size}] {image_path.name[:50]:50} | Similarity: {similarity:.6f}")
    
    if not similarities:
        print("\n❌ No se pudo calcular ninguna similitud")
        return None
    
    similarities = np.array(similarities)
    avg_sim = similarities.mean()
    
    print(f"\n{'='*70}")
    print(f"RESULTADO: Self-similarity promedio = {avg_sim:.6f}")
    print(f"{'='*70}")
    
    if avg_sim > 0.99:
        print("✅ EXCELENTE: Embeddings son determinísticos y consistentes")
    elif avg_sim > 0.95:
        print("✅ BUENO: Ligera variación (probablemente por normalización)")
    else:
        print("❌ PROBLEMA: Embeddings NO son consistentes")
        print("   → Revisar normalización o precisión numérica")
    
    print()
    return similarities


def test_train_train_similarity():
    """
    TEST 2: Train-Train Similarity
    
    Compara embeddings entre imágenes del train set.
    Esperado: Imágenes con defectos similares tienen alta similitud
    """
    print("="*70)
    print("TEST 2: TRAIN-TRAIN SIMILARITY (imágenes indexadas entre sí)")
    print("="*70 + "\n")
    
    # Cargar embeddings y metadata del train
    embeddings_path = Path("data/processed/embeddings/fullimages_dinov3/embeddings_fullimages_dinov3.npy")
    metadata_path = Path("data/processed/embeddings/fullimages_dinov3/metadata_fullimages.json")
    
    if not embeddings_path.exists() or not metadata_path.exists():
        print(f"❌ Archivos no encontrados")
        return None
    
    embeddings = np.load(embeddings_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"Embeddings cargados: {embeddings.shape}")
    print(f"Metadata cargada: {len(metadata)} entradas\n")
    
    # Agrupar por tipo de daño dominante
    damage_groups = {}
    for i, meta in enumerate(metadata):
        types = meta.get('defect_types', [])
        if not types:
            continue
        
        # Tipo más común
        dominant = Counter(types).most_common(1)[0][0]
        
        if dominant not in damage_groups:
            damage_groups[dominant] = []
        damage_groups[dominant].append((i, meta))
    
    print(f"Grupos de daños encontrados: {list(damage_groups.keys())}\n")
    
    # Calcular similitud intra-grupo vs inter-grupo
    print("Similitud INTRA-GRUPO (mismo tipo de daño):")
    print("-" * 70)
    
    intra_similarities = []
    
    for damage_type, items in sorted(damage_groups.items()):
        if len(items) < 2:
            continue
        
        # Tomar primeras 5 imágenes de este tipo
        indices = [item[0] for item in items[:5]]
        
        # Calcular similitud promedio entre ellas
        sims = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                sim = 1 - cosine(embeddings[indices[i]], embeddings[indices[j]])
                sims.append(sim)
        
        if sims:
            avg_sim = np.mean(sims)
            intra_similarities.extend(sims)
            print(f"  {damage_type:20} | Avg similarity: {avg_sim:.4f} ({len(sims)} pares)")
    
    print(f"\n{'='*70}")
    print(f"SIMILITUD INTRA-GRUPO PROMEDIO: {np.mean(intra_similarities):.4f}")
    print(f"{'='*70}")
    
    if np.mean(intra_similarities) > 0.7:
        print("✅ EXCELENTE: Imágenes con mismos defectos son muy similares")
    elif np.mean(intra_similarities) > 0.5:
        print("✅ BUENO: Imágenes con mismos defectos tienen similitud moderada")
    else:
        print("⚠️  BAJO: Imágenes con mismos defectos tienen baja similitud")
        print("   → Puede indicar alta variabilidad visual dentro de cada tipo")
    
    print()
    return intra_similarities


def test_test_train_retrieval():
    """
    TEST 3: Test-Train Retrieval Quality
    
    Toma imágenes del test set y busca en el índice FAISS.
    Esperado: Recupera imágenes con tipos de daño similares
    """
    print("="*70)
    print("TEST 3: TEST-TRAIN RETRIEVAL (calidad de búsqueda)")
    print("="*70 + "\n")
    
    from src.core.rag.retriever import DamageRAGRetriever
    
    # Cargar retriever
    retriever = DamageRAGRetriever(
        index_path=Path("outputs/vector_indices/fullimages_dinov3/indexhnswflat_fullimages.index"),
        metadata_path=Path("outputs/vector_indices/fullimages_dinov3/metadata_fullimages.pkl")
    )
    
    # Cargar test set
    test_manifest_path = Path("data/raw/train_test_split_8020/test/test_manifest.json")
    
    if not test_manifest_path.exists():
        print(f"❌ Test manifest no encontrado")
        return None
    
    with open(test_manifest_path) as f:
        test_manifest = json.load(f)
    
    # Tomar 10 imágenes del test
    embedder = DINOv3ViTLEmbedder()
    
    sample_size = 10
    sampled_test = test_manifest[:sample_size]
    
    print(f"Evaluando {len(sampled_test)} imágenes del test set\n")
    
    recall_scores = []
    
    for i, test_item in enumerate(sampled_test, 1):
        test_image_path = Path("data/raw/train_test_split_8020/test") / test_item['image']
        
        if not test_image_path.exists():
            continue
        
        # Ground truth
        gt_types = set(test_item['defect_distribution'].keys())
        
        # Generar embedding
        query_emb = embedder.generate_embedding(test_image_path, normalize=True)
        
        # Buscar Top-5
        results = retriever.search(query_emb, k=5)
        
        # Tipos recuperados
        retrieved_types = set()
        for r in results:
            retrieved_types.update(r.damage_type)
        
        # Calcular recall
        hits = len(gt_types & retrieved_types)
        recall = hits / len(gt_types) if gt_types else 0.0
        recall_scores.append(recall)
        
        status = "✅" if recall >= 0.5 else "⚠️" if recall > 0 else "❌"
        print(f"{status} [{i:2d}/{sample_size}] {test_image_path.name[:40]:40} | Recall: {recall:.2%}")
        print(f"     GT: {gt_types}")
        print(f"     Retrieved: {retrieved_types}")
        print()
    
    if not recall_scores:
        print("\n❌ No se pudo calcular recall")
        return None
    
    avg_recall = np.mean(recall_scores)
    
    print(f"{'='*70}")
    print(f"RECALL@5 PROMEDIO: {avg_recall:.2%}")
    print(f"{'='*70}")
    
    if avg_recall > 0.6:
        print("✅ EXCELENTE: Retrieval recupera tipos de daño correctos")
    elif avg_recall > 0.4:
        print("✅ BUENO: Retrieval tiene precisión aceptable")
    elif avg_recall > 0.2:
        print("⚠️  MODERADO: Retrieval necesita mejoras")
    else:
        print("❌ BAJO: Retrieval NO funciona correctamente")
        print("   → Revisar normalización de labels o similitud de embeddings")
    
    print()
    return recall_scores


def test_embedding_distribution():
    """
    TEST 4: Distribución de Embeddings
    
    Verifica que los embeddings tengan buena distribución en el espacio.
    """
    print("="*70)
    print("TEST 4: DISTRIBUCIÓN DE EMBEDDINGS")
    print("="*70 + "\n")
    
    embeddings_path = Path("data/processed/embeddings/fullimages_dinov3/embeddings_fullimages_dinov3.npy")
    
    if not embeddings_path.exists():
        print(f"❌ Embeddings no encontrados")
        return None
    
    embeddings = np.load(embeddings_path)
    
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}\n")
    
    # Estadísticas
    norms = np.linalg.norm(embeddings, axis=1)
    
    print("Estadísticas de normas:")
    print(f"  - Media: {norms.mean():.4f}")
    print(f"  - Std: {norms.std():.4f}")
    print(f"  - Min: {norms.min():.4f}")
    print(f"  - Max: {norms.max():.4f}")
    
    # Verificar normalización
    if np.abs(norms.mean() - 1.0) < 0.01:
        print("\n✅ Embeddings están normalizados correctamente")
    else:
        print(f"\n⚠️  Embeddings NO están normalizados (norma media = {norms.mean():.4f})")
    
    # Distribución de valores
    print(f"\nDistribución de valores:")
    print(f"  - Media de componentes: {embeddings.mean():.4f}")
    print(f"  - Std de componentes: {embeddings.std():.4f}")
    print(f"  - Min valor: {embeddings.min():.4f}")
    print(f"  - Max valor: {embeddings.max():.4f}")
    
    # Diversidad (distancia promedio entre vectores aleatorios)
    print(f"\nDiversidad del espacio:")
    n_samples = min(100, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    
    distances = []
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            dist = cosine(embeddings[indices[i]], embeddings[indices[j]])
            distances.append(dist)
    
    avg_dist = np.mean(distances)
    print(f"  - Distancia coseno promedio: {avg_dist:.4f}")
    
    if avg_dist > 0.3:
        print("  ✅ Buena diversidad: Embeddings bien distribuidos en el espacio")
    else:
        print("  ⚠️  Baja diversidad: Embeddings muy similares entre sí")
    
    print()
    return embeddings


def run_all_tests():
    """Ejecuta todos los tests de validación"""
    
    print("\n" + "="*70)
    print("🔬 DIAGNÓSTICO COMPLETO: FULL IMAGES CONSISTENCY")
    print("="*70 + "\n")
    
    results = {}
    
    # Test 1: Self-similarity
    print("Ejecutando TEST 1...\n")
    results['self_similarity'] = test_self_similarity()
    
    # Test 2: Train-Train similarity
    print("\n" + "="*70 + "\n")
    print("Ejecutando TEST 2...\n")
    results['train_train_similarity'] = test_train_train_similarity()
    
    # Test 3: Test-Train retrieval
    print("\n" + "="*70 + "\n")
    print("Ejecutando TEST 3...\n")
    results['retrieval_recall'] = test_test_train_retrieval()
    
    # Test 4: Embedding distribution
    print("\n" + "="*70 + "\n")
    print("Ejecutando TEST 4...\n")
    results['embeddings'] = test_embedding_distribution()
    
    # Resumen final
    print("\n" + "="*70)
    print("📊 RESUMEN FINAL")
    print("="*70 + "\n")
    
    if results['self_similarity'] is not None:
        avg_self = np.mean(results['self_similarity'])
        status = "✅" if avg_self > 0.99 else "⚠️"
        print(f"{status} Self-similarity: {avg_self:.4f}")
    
    if results['train_train_similarity'] is not None:
        avg_intra = np.mean(results['train_train_similarity'])
        status = "✅" if avg_intra > 0.5 else "⚠️"
        print(f"{status} Intra-group similarity: {avg_intra:.4f}")
    
    if results['retrieval_recall'] is not None:
        avg_recall = np.mean(results['retrieval_recall'])
        status = "✅" if avg_recall > 0.4 else "⚠️"
        print(f"{status} Retrieval Recall@5: {avg_recall:.2%}")
    
    print(f"\n{'='*70}")
    print("🎯 CONCLUSIÓN")
    print(f"{'='*70}")
    
    if results['retrieval_recall'] is not None:
        avg_recall = np.mean(results['retrieval_recall'])
        
        if avg_recall > 0.5:
            print("✅ SISTEMA FUNCIONANDO CORRECTAMENTE")
            print("   → Full images strategy está funcionando bien")
            print("   → Mejora significativa vs crops (recall 0.0 → {:.0%})".format(avg_recall))
        elif avg_recall > 0.3:
            print("⚠️  SISTEMA CON RENDIMIENTO MODERADO")
            print("   → Considerar hybrid embeddings (visual + textual)")
            print("   → Verificar taxonomía normalizer")
        else:
            print("❌ SISTEMA NECESITA MEJORAS")
            print("   → Revisar similitud de embeddings")
            print("   → Verificar normalización de labels")
    
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
# diagnosis/test_taxonomy_alignment.py

TRAIN_TAXONOMY = {
    "1": "surface_scratch",
    "2": "dent", 
    "3": "paint_peeling",
    "4": "deep_scratch",
    "5": "crack",
    "6": "missing_part",
    "7": "missing_accessory",
    "8": "misaligned_part"
}

TEST_TAXONOMY = [
    "Scratch", "Dent", "Degraded varnish", "Crack",
    "Fractured part", "Missing part", "Deviated part",
    "No damage", "Unknown"
]

def analyze_taxonomy_coverage():
    """
    Analiza qué porcentaje de crops en el índice FAISS tienen 
    correspondencia directa con la taxonomía de evaluación
    """
    import pickle
    from pathlib import Path
    from collections import Counter
    
    # Cargar metadata del índice
    metadata_path = Path("outputs/vector_indices/train_set_dinov3/metadata_clustered.pkl")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Contar distribución de tipos en train
    train_types = []
    for meta in metadata:
        dtype = meta.get('dominant_type', meta.get('damage_type', 'unknown'))
        train_types.append(dtype)
    
    train_distribution = Counter(train_types)
    
    print("="*70)
    print("ANÁLISIS DE COBERTURA TAXONÓMICA")
    print("="*70)
    print(f"\nTotal crops indexados: {len(metadata)}\n")
    
    # Mapeo propuesto (basado en semántica)
    mapping = {
        "surface_scratch": "Scratch",
        "deep_scratch": "Scratch",
        "dent": "Dent",
        "paint_peeling": "Degraded varnish",
        "crack": "Crack",
        "missing_part": "Missing part",
        "missing_accessory": "Missing part",
        "misaligned_part": "Deviated part"
    }
    
    # Calcular cobertura
    print("Distribución Train → Test:")
    print("-"*70)
    
    total_covered = 0
    total_uncovered = 0
    
    for train_type, count in sorted(train_distribution.items(), key=lambda x: -x[1]):
        test_type = mapping.get(train_type, "❌ SIN MAPEO")
        coverage = "✅" if train_type in mapping else "❌"
        
        if train_type in mapping:
            total_covered += count
        else:
            total_uncovered += count
        
        print(f"{coverage} {train_type:20} → {test_type:20} ({count:4} crops, {count/len(metadata)*100:5.1f}%)")
    
    print("-"*70)
    print(f"\nCobertura total: {total_covered}/{len(metadata)} crops ({total_covered/len(metadata)*100:.1f}%)")
    print(f"Sin cobertura:   {total_uncovered}/{len(metadata)} crops ({total_uncovered/len(metadata)*100:.1f}%)")
    
    # Analizar tipos que faltan en train
    print("\n" + "="*70)
    print("TIPOS EN TEST QUE NO EXISTEN EN TRAIN")
    print("="*70)
    
    mapped_test_types = set(mapping.values())
    missing_in_train = set(TEST_TAXONOMY) - mapped_test_types - {"No damage", "Unknown"}
    
    if missing_in_train:
        print("\n⚠️  Tipos de test SIN representación en train:")
        for mtype in missing_in_train:
            print(f"   - {mtype}")
        print("\n💡 El RAG NUNCA podrá recuperar ejemplos para estos tipos")
    else:
        print("\n✅ Todos los tipos de test están cubiertos por el mapping")
    
    return mapping, train_distribution

if __name__ == "__main__":
    mapping, distribution = analyze_taxonomy_coverage()
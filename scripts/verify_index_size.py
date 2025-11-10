# scripts/verify_index_size.py (NUEVO)
import faiss
import pickle

index = faiss.read_index("outputs/vector_indices/train_set_dinov3/indexhnswflat_clustered.index")
with open("outputs/vector_indices/train_set_dinov3/metadata_clustered.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"Vectores: {index.ntotal}")
print(f"Metadata: {len(metadata)}")

# Distribución de tipos
from collections import Counter
types = [m.get('dominant_type', 'unknown') for m in metadata]
print("\nDistribución:")
for dtype, count in Counter(types).most_common():
    print(f"  {dtype}: {count}")
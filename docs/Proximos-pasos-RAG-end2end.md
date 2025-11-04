# 🎯 GUÍA COMPLETA: EVALUACIÓN RAG CORRECTA

## ❌ Lo Que Estabas Haciendo (Incorrecto)

```
Dataset → Embeddings → Índice FAISS
                ↓
        MISMOS datos como queries
                ↓
        Recall@5 = 100% (trampa!)
```

**Problema**: Data leakage - evalúas con los mismos datos que usaste para entrenar.

---

## ✅ Evaluación RAG Correcta

### Pipeline Completo

```
Dataset Completo (2,711 imágenes)
        ↓
    SPLIT 80/20
        ↓
   ┌────────┴────────┐
   ↓                 ↓
TRAIN (80%)      TEST (20%)
2,168 imgs       543 imgs (NUNCA VISTAS)
   ↓                 ↓
Generar          Usar como
Embeddings       queries
   ↓
Índice FAISS
   ↓
Retrieval ← Test images (queries)
   ↓
RAG Context
   ↓
Qwen3VL genera respuesta
   ↓
Evaluar vs Ground Truth
```

---

## 🚀 PLAN DE ACCIÓN (3 Horas)

### Paso 1: Train/Test Split (15 min)

```bash
# Dividir dataset en 80% train / 20% test
python scripts/06_split_train_test.py \
    --source /home/carlos/Escritorio/Proyectos-Minsait/Mapfre/carpetas-datos/jsons_segmentacion_jsonsfinales \
    --output data/raw/train_test_split \
    --test-size 0.20 \
    --seed 42
```

**Output esperado:**
```
TRAIN: 2,168 imágenes (80%)
TEST:  543 imágenes (20%)
```

**IMPORTANTE:** 
- Train set → generar embeddings e índice
- Test set → NUNCA tocar hasta la evaluación final

---

### Paso 2: Procesar SOLO Train Set (1.5 horas)

```bash
# 2.1 Generar crops del TRAIN set
python scripts/02_generate_clustered_crops.py \
    --dataset data/raw/train_test_split/train \
    --output data/processed/crops/train_set

# 2.2 Generar embeddings del TRAIN set
python scripts/03_generate_embeddings_dinov3.py \
    --metadata data/processed/crops/train_set/metadata/clustered_crops_metadata.json \
    --output data/processed/embeddings/train_set

# 2.3 Construir índice FAISS del TRAIN set
python scripts/04_build_faiss_index.py \
    --embeddings data/processed/embeddings/train_set/embeddings_dinov3_vitl.npy \
    --metadata data/processed/embeddings/train_set/enriched_crops_metadata_dinov3_vitl.json \
    --output outputs/vector_indices/train_set
```

---

### Paso 3: Evaluación RAG End-to-End (1 hora)

```bash
# Evaluar con TEST set (imágenes NUNCA vistas)
python scripts/07_evaluate_rag_end_to_end.py \
    --test-set data/raw/train_test_split/test \
    --index outputs/vector_indices/train_set/indexhnswflat_clustered.index \
    --metadata outputs/vector_indices/train_set/metadata_clustered.pkl \
    --k 5 \
    --max-cases 50  # Primero probar con 50 imágenes
```

**Esto SÍ es una evaluación real:**
- ✅ Imágenes test NUNCA vistas
- ✅ Genera embeddings on-the-fly
- ✅ Retrieval en índice del train set
- ✅ Qwen3VL genera respuesta con contexto RAG
- ✅ Evalúa calidad vs ground truth

---

## 📊 Métricas RAG Correctas

### Retrieval Metrics
- **Recall@k**: ¿Se recuperan los tipos de daño correctos?
- **MRR**: ¿En qué posición aparece el tipo correcto?
- **Precision@k**: ¿Qué % de retrieved son relevantes?

### Generation Metrics
- **Answer Correctness**: ¿La respuesta identifica los daños correctos?
- **Faithfulness**: ¿La respuesta usa el contexto RAG?
- **Hallucination Rate**: ¿Inventa daños que no existen?

### Performance Metrics
- **Retrieval Time**: Tiempo de búsqueda FAISS
- **Generation Time**: Tiempo Qwen3VL
- **Total Latency**: End-to-end

---

## 🎯 Resultados Esperados

### ✅ Con Evaluación Correcta

```
Test Set (543 imágenes nunca vistas):
  - Recall@5: 75-85% (realista)
  - MRR: 0.65-0.75 (realista)
  - Answer Correctness: 70-80%
```

**Estos son resultados REALES** que reflejan performance en producción.

### ❌ Lo Que Tenías Antes (Incorrecto)

```
Same images (1,024 crops):
  - Recall@5: 100% (trampa - mismo dato)
  - MRR: 1.0 (trampa - busca exactamente sí mismo)
```

---

## 💡 Ejemplo Concreto

### Imagen de Test (NUNCA vista)

```
test/zona5_1234_original.jpg
Ground truth:
  - 3x surface_scratch
  - 1x dent
  - 1x paint_peeling
```

### Pipeline RAG

**1. Genera embedding** (DINOv3)
```python
query_emb = embedder.generate_embedding("test/zona5_1234_original.jpg")
# shape: (1024,)
```

**2. Retrieval FAISS** (busca en train set)
```python
results = retriever.search(query_emb, k=5)
# Recupera 5 crops más similares del TRAIN set
```

**3. Construye contexto RAG**
```
## Ejemplos Similares:

### Ejemplo 1 (similitud: 92%):
- Tipo: surface_scratch
- Zona: hood_center
- Imagen: train/zona2_5678_crop_045.jpg

### Ejemplo 2 (similitud: 88%):
- Tipo: dent
- Zona: front_left
...
```

**4. Genera respuesta con Qwen3VL**
```json
{
  "damages": [
    {"type": "surface_scratch", "location": "hood", "count": 3},
    {"type": "dent", "location": "door_left", "count": 1},
    {"type": "paint_peeling", "location": "bumper", "count": 1}
  ]
}
```

**5. Evalúa vs Ground Truth**
```
Recall@5: 100% (encontró surface_scratch, dent, paint_peeling)
Answer Correctness: 100% (identificó todos correctamente)
```

---

## 🔄 Comparación: Antes vs Después

| Aspecto | ❌ Antes (Incorrecto) | ✅ Después (Correcto) |
|---------|----------------------|----------------------|
| **Train/Test Split** | No | Sí (80/20) |
| **Queries** | Mismo train set | Test set (no visto) |
| **Recall@5** | 100% (trampa) | 75-85% (realista) |
| **Genera respuesta VLM** | No | Sí (Qwen3VL) |
| **Evalúa respuesta** | No | Sí (vs ground truth) |
| **Refleja producción** | No | Sí |

---

## ✨ Bonus: Evaluación Humana

Para máxima validez, añade evaluación humana:

```bash
# Genera reporte con imágenes
python scripts/08_generate_visual_report.py \
    --results outputs/rag_evaluation/rag_evaluation_results.json \
    --output outputs/rag_evaluation/visual_report.html
```

Luego revisa manualmente 20-30 casos:
- ¿La respuesta es correcta?
- ¿El contexto RAG fue útil?
- ¿Hay hallucinations?

---

## 📌 Checklist Final

Antes de confiar en resultados RAG:

- [ ] Train/test split hecho (80/20)
- [ ] Índice FAISS generado SOLO del train set
- [ ] Test set NUNCA visto durante training
- [ ] Evaluación usa imágenes test como queries
- [ ] Pipeline genera embeddings on-the-fly
- [ ] Qwen3VL genera respuestas con contexto RAG
- [ ] Métricas calculadas vs ground truth
- [ ] Recall@5 es realista (70-85%, no 100%)

Si marcaste todo ✅, tienes una **evaluación RAG válida** 🎉

---

## 🚀 Ejecución Rápida

```bash
# Todo en un script
./scripts/run_full_rag_evaluation.sh
```

O paso a paso:

```bash
# 1. Split (15 min)
python scripts/06_split_train_test.py

# 2. Train pipeline (1.5 hrs)
python scripts/02_generate_clustered_crops.py --dataset data/raw/train_test_split/train
python scripts/03_generate_embeddings_dinov3.py --metadata data/processed/crops/train_set/metadata/...
python scripts/04_build_faiss_index.py --embeddings data/processed/embeddings/train_set/...

# 3. Evaluate (1 hr)
python scripts/07_evaluate_rag_end_to_end.py --test-set data/raw/train_test_split/test
```
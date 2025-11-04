# 📊 RAG MULTIMODAL - REPORTE COMPARATIVO

**Fecha**: 2025-11-04 16:16:44

**Configuraciones evaluadas**: 6

---

## 🎯 Resumen Ejecutivo

### 🏆 Mejores Configuraciones

- **Mejor Recall@5**: `dinov3_high` (100.00%)
- **Mejor MRR**: `dinov3_high` (1.000)
- **Más Rápido**: `qwen3_low` (0.0ms)

### 📊 Dispersión de Resultados

- **Rango Recall@5**: 95.61% - 100.00% (Δ = 4.39%)
- **Rango MRR**: 0.677 - 1.000
- **Rango Latencia**: 0.0ms - 0.1ms


---

## 📈 Tabla Comparativa de Métricas

| Config | Embedding | Density | R@1 | R@3 | R@5 | MRR | Lat. (ms) |
|--------|-----------|---------|-----|-----|-----|-----|-----------|
| dinov3_high | dinov3 | high | 100.00% | 100.00% | 100.00% | 1.000 | 0.1 |
| qwen3_high | qwen3 | high | 100.00% | 100.00% | 100.00% | 1.000 | 0.0 |
| qwen3_low | qwen3 | low | 94.63% | 97.66% | 98.83% | 0.963 | 0.0 |
| qwen3_medium | qwen3 | medium | 96.19% | 98.54% | 98.63% | 0.973 | 0.0 |
| dinov3_medium | dinov3 | medium | 89.16% | 95.41% | 96.00% | 0.923 | 0.0 |
| dinov3_low | dinov3 | low | 49.51% | 86.43% | 95.61% | 0.677 | 0.0 |

**Leyenda**:
- R@k: Recall@k (% de queries donde el tipo correcto está en top-k)
- MRR: Mean Reciprocal Rank (promedio de 1/rank del primer resultado correcto)
- Lat.: Latencia promedio total (retrieval + generación)


---

## 🔬 Análisis por Tipo de Embedding

### DINOv3-ViT-L/16 (1024 dims)

- **Recall@5 promedio**: 97.20%
- **MRR promedio**: 0.867
- **Latencia promedio**: 0.0ms

### Qwen3VL + Sentence-BERT (384 dims)

- **Recall@5 promedio**: 99.15%
- **MRR promedio**: 0.979
- **Latencia promedio**: 0.0ms

### 🔍 Comparación

✅ **Qwen3VL** supera a DINOv3 en Recall@5 por **1.95%**
⚡ **Qwen3VL** es **1.49x** más rápido


---

## 📦 Análisis por Densidad de Dataset

### High Density

- **Configuraciones**: 2
- **Recall@5 promedio**: 100.00%
- **MRR promedio**: 1.000

### Medium Density

- **Configuraciones**: 2
- **Recall@5 promedio**: 97.31%
- **MRR promedio**: 0.948

### Low Density

- **Configuraciones**: 2
- **Recall@5 promedio**: 97.22%
- **MRR promedio**: 0.820


---

## ⚡ Análisis de Latencia

### ⚡ Retrieval (Top-3 más rápidos)

- `qwen3_low`: 0.0ms
- `dinov3_low`: 0.0ms
- `qwen3_medium`: 0.0ms

### 🤖 Generación VLM (Top-3 más rápidos)

- `dinov3_high`: 0.0ms
- `dinov3_medium`: 0.0ms
- `dinov3_low`: 0.0ms

### 🏁 Latencia Total (Top-3 más rápidos)

- `qwen3_low`: 0.0ms
- `dinov3_low`: 0.0ms
- `qwen3_medium`: 0.0ms


---

## 💡 Recomendaciones

### 🎯 Uso Recomendado por Caso

**1. Máxima Precisión (Recall)**
   - Configuración: `dinov3_high`
   - Recall@5: 100.00%
   - Uso: Aplicaciones donde la precisión es crítica

**2. Máxima Velocidad**
   - Configuración: `qwen3_low`
   - Latencia: 0.0ms
   - Uso: Aplicaciones en tiempo real

**3. Balance Calidad/Velocidad**
   - Configuración: `qwen3_high`
   - Recall@5: 100.00%
   - Latencia: 0.0ms


---

## 🗂️ Distribución de Tipos de Daño Recuperados

### dinov3_high

- `surface_scratch`: 4938 (96.4%)
- `dent`: 77 (1.5%)
- `misaligned_part`: 38 (0.7%)
- `missing_accessory`: 28 (0.5%)
- `missing_part`: 15 (0.3%)

### dinov3_medium

- `surface_scratch`: 4703 (91.9%)
- `dent`: 183 (3.6%)
- `missing_part`: 73 (1.4%)
- `deep_scratch`: 66 (1.3%)
- `paint_peeling`: 58 (1.1%)

### dinov3_low

- `surface_scratch`: 1937 (37.8%)
- `dent`: 1359 (26.5%)
- `paint_peeling`: 830 (16.2%)
- `crack`: 758 (14.8%)
- `misaligned_part`: 101 (2.0%)

### qwen3_high

- `surface_scratch`: 4882 (95.4%)
- `dent`: 101 (2.0%)
- `misaligned_part`: 51 (1.0%)
- `missing_accessory`: 36 (0.7%)
- `crack`: 22 (0.4%)

### qwen3_medium

- `surface_scratch`: 4829 (94.3%)
- `dent`: 114 (2.2%)
- `deep_scratch`: 67 (1.3%)
- `missing_part`: 66 (1.3%)
- `missing_accessory`: 21 (0.4%)

### qwen3_low

- `surface_scratch`: 4697 (91.7%)
- `dent`: 139 (2.7%)
- `deep_scratch`: 109 (2.1%)
- `paint_peeling`: 59 (1.2%)
- `crack`: 51 (1.0%)

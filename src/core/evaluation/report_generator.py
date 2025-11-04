# src/core/evaluation/report_generator.py

from pathlib import Path
import json
from typing import Dict, List
from dataclasses import asdict
import numpy as np

class RAGReportGenerator:
    """
    Genera reportes comparativos entre sistemas RAG
    
    Funcionalidades:
    1. Tablas comparativas de métricas
    2. Análisis estadístico por configuración
    3. Mejores/peores casos por sistema
    4. Recomendaciones basadas en resultados
    """
    
    def __init__(self, output_dir: Path = Path("outputs/evaluation/reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comparison_report(
        self,
        metrics_by_config: Dict,
        output_name: str = "comparison_report"
    ) -> Path:
        """
        Genera reporte comparativo en formato Markdown
        """
        report_lines = []
        
        # Header
        report_lines.append("# 📊 RAG MULTIMODAL - REPORTE COMPARATIVO\n")
        report_lines.append(f"**Fecha**: {self._get_timestamp()}\n")
        report_lines.append(f"**Configuraciones evaluadas**: {len(metrics_by_config)}\n")
        report_lines.append("---\n")
        
        # Executive Summary
        report_lines.append("## 🎯 Resumen Ejecutivo\n")
        report_lines.extend(self._generate_executive_summary(metrics_by_config))
        report_lines.append("\n---\n")
        
        # Tabla comparativa principal
        report_lines.append("## 📈 Tabla Comparativa de Métricas\n")
        report_lines.extend(self._generate_metrics_table(metrics_by_config))
        report_lines.append("\n---\n")
        
        # Análisis por tipo de embedding
        report_lines.append("## 🔬 Análisis por Tipo de Embedding\n")
        report_lines.extend(self._analyze_by_embedding_type(metrics_by_config))
        report_lines.append("\n---\n")
        
        # Análisis por densidad de dataset
        report_lines.append("## 📦 Análisis por Densidad de Dataset\n")
        report_lines.extend(self._analyze_by_density(metrics_by_config))
        report_lines.append("\n---\n")
        
        # Análisis de latencia
        report_lines.append("## ⚡ Análisis de Latencia\n")
        report_lines.extend(self._analyze_latency(metrics_by_config))
        report_lines.append("\n---\n")
        
        # Recomendaciones
        report_lines.append("## 💡 Recomendaciones\n")
        report_lines.extend(self._generate_recommendations(metrics_by_config))
        report_lines.append("\n---\n")
        
        # Distribución de tipos de daño
        report_lines.append("## 🗂️ Distribución de Tipos de Daño Recuperados\n")
        report_lines.extend(self._analyze_damage_distribution(metrics_by_config))
        
        # Guardar reporte
        report_path = self.output_dir / f"{output_name}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n📄 Reporte generado: {report_path}\n")
        return report_path
    
    def _generate_executive_summary(self, metrics: Dict) -> List[str]:
        """Genera resumen ejecutivo con mejores configuraciones"""
        lines = []
        
        # Ordenar por Recall@5
        sorted_by_recall = sorted(
            metrics.items(),
            key=lambda x: x[1].recall_at_5,
            reverse=True
        )
        
        # Ordenar por MRR
        sorted_by_mrr = sorted(
            metrics.items(),
            key=lambda x: x[1].mrr,
            reverse=True
        )
        
        # Ordenar por latencia (menor es mejor)
        sorted_by_speed = sorted(
            metrics.items(),
            key=lambda x: x[1].avg_total_time_ms
        )
        
        best_recall = sorted_by_recall[0]
        best_mrr = sorted_by_mrr[0]
        fastest = sorted_by_speed[0]
        
        lines.append(f"### 🏆 Mejores Configuraciones\n")
        lines.append(f"- **Mejor Recall@5**: `{best_recall[0]}` ({best_recall[1].recall_at_5:.2%})")
        lines.append(f"- **Mejor MRR**: `{best_mrr[0]}` ({best_mrr[1].mrr:.3f})")
        lines.append(f"- **Más Rápido**: `{fastest[0]}` ({fastest[1].avg_total_time_ms:.1f}ms)\n")
        
        # Diferencia entre mejor y peor
        worst_recall = sorted_by_recall[-1]
        diff = best_recall[1].recall_at_5 - worst_recall[1].recall_at_5
        
        lines.append(f"### 📊 Dispersión de Resultados\n")
        lines.append(f"- **Rango Recall@5**: {worst_recall[1].recall_at_5:.2%} - {best_recall[1].recall_at_5:.2%} (Δ = {diff:.2%})")
        lines.append(f"- **Rango MRR**: {sorted_by_mrr[-1][1].mrr:.3f} - {sorted_by_mrr[0][1].mrr:.3f}")
        lines.append(f"- **Rango Latencia**: {sorted_by_speed[0][1].avg_total_time_ms:.1f}ms - {sorted_by_speed[-1][1].avg_total_time_ms:.1f}ms\n")
        
        return lines
    
    def _generate_metrics_table(self, metrics: Dict) -> List[str]:
        """Genera tabla markdown con todas las métricas"""
        lines = []
        
        # Header
        lines.append("| Config | Embedding | Density | R@1 | R@3 | R@5 | MRR | Lat. (ms) |")
        lines.append("|--------|-----------|---------|-----|-----|-----|-----|-----------|")
        
        # Ordenar por Recall@5
        sorted_metrics = sorted(
            metrics.items(),
            key=lambda x: x[1].recall_at_5,
            reverse=True
        )
        
        for config_name, m in sorted_metrics:
            # Extraer embedding type y density del nombre
            parts = config_name.split('_')
            emb_type = parts[0]  # 'dinov3' o 'qwen3vl'
            density = parts[1] if len(parts) > 1 else 'unknown'
            
            lines.append(
                f"| {config_name} | {emb_type} | {density} | "
                f"{m.recall_at_1:.2%} | {m.recall_at_3:.2%} | {m.recall_at_5:.2%} | "
                f"{m.mrr:.3f} | {m.avg_total_time_ms:.1f} |"
            )
        
        lines.append("")
        
        # Leyenda
        lines.append("**Leyenda**:")
        lines.append("- R@k: Recall@k (% de queries donde el tipo correcto está en top-k)")
        lines.append("- MRR: Mean Reciprocal Rank (promedio de 1/rank del primer resultado correcto)")
        lines.append("- Lat.: Latencia promedio total (retrieval + generación)\n")
        
        return lines
    
    def _analyze_by_embedding_type(self, metrics: Dict) -> List[str]:
        """Compara DINOv3 vs Qwen3VL"""
        lines = []
        
        # Separar por tipo de embedding
        dinov3_metrics = {k: v for k, v in metrics.items() if 'dinov3' in k}
        qwen_metrics = {k: v for k, v in metrics.items() if 'qwen3' in k}
        
        # Promedios
        dinov3_avg_recall5 = np.mean([m.recall_at_5 for m in dinov3_metrics.values()])
        qwen_avg_recall5 = np.mean([m.recall_at_5 for m in qwen_metrics.values()])
        
        dinov3_avg_mrr = np.mean([m.mrr for m in dinov3_metrics.values()])
        qwen_avg_mrr = np.mean([m.mrr for m in qwen_metrics.values()])
        
        dinov3_avg_latency = np.mean([m.avg_total_time_ms for m in dinov3_metrics.values()])
        qwen_avg_latency = np.mean([m.avg_total_time_ms for m in qwen_metrics.values()])
        
        lines.append("### DINOv3-ViT-L/16 (1024 dims)\n")
        lines.append(f"- **Recall@5 promedio**: {dinov3_avg_recall5:.2%}")
        lines.append(f"- **MRR promedio**: {dinov3_avg_mrr:.3f}")
        lines.append(f"- **Latencia promedio**: {dinov3_avg_latency:.1f}ms\n")
        
        lines.append("### Qwen3VL + Sentence-BERT (384 dims)\n")
        lines.append(f"- **Recall@5 promedio**: {qwen_avg_recall5:.2%}")
        lines.append(f"- **MRR promedio**: {qwen_avg_mrr:.3f}")
        lines.append(f"- **Latencia promedio**: {qwen_avg_latency:.1f}ms\n")
        
        # Comparación
        lines.append("### 🔍 Comparación\n")
        
        if dinov3_avg_recall5 > qwen_avg_recall5:
            diff = dinov3_avg_recall5 - qwen_avg_recall5
            lines.append(f"✅ **DINOv3** supera a Qwen3VL en Recall@5 por **{diff:.2%}**")
        else:
            diff = qwen_avg_recall5 - dinov3_avg_recall5
            lines.append(f"✅ **Qwen3VL** supera a DINOv3 en Recall@5 por **{diff:.2%}**")
        
        if dinov3_avg_latency < qwen_avg_latency:
            speedup = qwen_avg_latency / dinov3_avg_latency
            lines.append(f"⚡ **DINOv3** es **{speedup:.2f}x** más rápido")
        else:
            speedup = dinov3_avg_latency / qwen_avg_latency
            lines.append(f"⚡ **Qwen3VL** es **{speedup:.2f}x** más rápido")
        
        lines.append("")
        
        return lines
    
    def _analyze_by_density(self, metrics: Dict) -> List[str]:
        """Analiza rendimiento por densidad de dataset"""
        lines = []
        
        # Separar por densidad
        high_metrics = {k: v for k, v in metrics.items() if 'high' in k}
        medium_metrics = {k: v for k, v in metrics.items() if 'medium' in k}
        low_metrics = {k: v for k, v in metrics.items() if 'low' in k}
        
        densities = {
            'High Density': high_metrics,
            'Medium Density': medium_metrics,
            'Low Density': low_metrics
        }
        
        for density_name, density_metrics in densities.items():
            if not density_metrics:
                continue
            
            avg_recall5 = np.mean([m.recall_at_5 for m in density_metrics.values()])
            avg_mrr = np.mean([m.mrr for m in density_metrics.values()])
            
            lines.append(f"### {density_name}\n")
            lines.append(f"- **Configuraciones**: {len(density_metrics)}")
            lines.append(f"- **Recall@5 promedio**: {avg_recall5:.2%}")
            lines.append(f"- **MRR promedio**: {avg_mrr:.3f}\n")
        
        return lines
    
    def _analyze_latency(self, metrics: Dict) -> List[str]:
        """Analiza tiempos de respuesta"""
        lines = []
        
        # Extraer tiempos
        retrieval_times = [(k, m.avg_retrieval_time_ms) for k, m in metrics.items()]
        generation_times = [(k, m.avg_generation_time_ms) for k, m in metrics.items()]
        total_times = [(k, m.avg_total_time_ms) for k, m in metrics.items()]
        
        # Ordenar
        retrieval_times.sort(key=lambda x: x[1])
        generation_times.sort(key=lambda x: x[1])
        total_times.sort(key=lambda x: x[1])
        
        lines.append("### ⚡ Retrieval (Top-3 más rápidos)\n")
        for config, time_ms in retrieval_times[:3]:
            lines.append(f"- `{config}`: {time_ms:.1f}ms")
        lines.append("")
        
        lines.append("### 🤖 Generación VLM (Top-3 más rápidos)\n")
        for config, time_ms in generation_times[:3]:
            lines.append(f"- `{config}`: {time_ms:.1f}ms")
        lines.append("")
        
        lines.append("### 🏁 Latencia Total (Top-3 más rápidos)\n")
        for config, time_ms in total_times[:3]:
            lines.append(f"- `{config}`: {time_ms:.1f}ms")
        lines.append("")
        
        return lines
    
    def _analyze_damage_distribution(self, metrics: Dict) -> List[str]:
        """Analiza distribución de tipos de daño recuperados"""
        lines = []
        
        for config_name, m in metrics.items():
            lines.append(f"### {config_name}\n")
            
            # Ordenar por frecuencia
            sorted_types = sorted(
                m.damage_type_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            total = sum(m.damage_type_distribution.values())
            
            for damage_type, count in sorted_types[:5]:  # Top-5
                percentage = (count / total * 100) if total > 0 else 0
                lines.append(f"- `{damage_type}`: {count} ({percentage:.1f}%)")
            
            lines.append("")
        
        return lines
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Genera recomendaciones basadas en resultados"""
        lines = []
        
        # Mejor overall
        best_overall = max(metrics.items(), key=lambda x: x[1].recall_at_5)
        
        # Mejor por velocidad
        fastest = min(metrics.items(), key=lambda x: x[1].avg_total_time_ms)
        
        lines.append("### 🎯 Uso Recomendado por Caso\n")
        
        lines.append(f"**1. Máxima Precisión (Recall)**")
        lines.append(f"   - Configuración: `{best_overall[0]}`")
        lines.append(f"   - Recall@5: {best_overall[1].recall_at_5:.2%}")
        lines.append(f"   - Uso: Aplicaciones donde la precisión es crítica\n")
        
        lines.append(f"**2. Máxima Velocidad**")
        lines.append(f"   - Configuración: `{fastest[0]}`")
        lines.append(f"   - Latencia: {fastest[1].avg_total_time_ms:.1f}ms")
        lines.append(f"   - Uso: Aplicaciones en tiempo real\n")
        
        # Balance
        balance_scores = {
            k: (m.recall_at_5 * 0.7 - m.avg_total_time_ms / 1000 * 0.3)
            for k, m in metrics.items()
        }
        best_balance = max(balance_scores.items(), key=lambda x: x[1])
        
        lines.append(f"**3. Balance Calidad/Velocidad**")
        lines.append(f"   - Configuración: `{best_balance[0]}`")
        lines.append(f"   - Recall@5: {metrics[best_balance[0]].recall_at_5:.2%}")
        lines.append(f"   - Latencia: {metrics[best_balance[0]].avg_total_time_ms:.1f}ms\n")
        
        return lines
    
    def _get_timestamp(self) -> str:
        """Retorna timestamp formateado"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_detailed_results(
        self,
        results_by_config: Dict,
        output_name: str = "detailed_results"
    ) -> Path:
        """
        Genera reporte detallado con ejemplos de queries
        """
        lines = []
        
        lines.append("# 🔍 RAG MULTIMODAL - RESULTADOS DETALLADOS\n")
        lines.append(f"**Fecha**: {self._get_timestamp()}\n")
        lines.append("---\n")
        
        # Para cada configuración, mostrar ejemplos
        for config_name, results in results_by_config.items():
            lines.append(f"## {config_name}\n")
            
            # Mostrar primeras 3 queries como ejemplo
            for i, result in enumerate(results[:3], 1):
                lines.append(f"### Query {i}: {result.query_id}\n")
                lines.append(f"**Tipo esperado**: `{result.expected_damage_type}`\n")
                lines.append(f"**Top-5 Recuperados**:")
                
                for j, (dtype, dist) in enumerate(zip(result.retrieved_damage_types, result.retrieved_distances), 1):
                    match = "✅" if dtype == result.expected_damage_type else "❌"
                    lines.append(f"{j}. {match} `{dtype}` (dist={dist:.4f})")
                
                lines.append(f"\n**Tiempos**:")
                lines.append(f"- Retrieval: {result.retrieval_time_ms:.1f}ms")
                lines.append(f"- Generación: {result.generation_time_ms:.1f}ms\n")
                
                if result.generated_answer:
                    lines.append("**Respuesta Generada**:")
                    lines.append(f"```\n{result.generated_answer}\n```\n")
                
                lines.append("---\n")
        
        # Guardar
        report_path = self.output_dir / f"{output_name}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"📄 Reporte detallado: {report_path}\n")
        return report_path
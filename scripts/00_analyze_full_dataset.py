# scripts/00_analyze_full_dataset.py

#!/usr/bin/env python3
"""
📊 ANÁLISIS EXHAUSTIVO DEL DATASET COMPLETO

Analiza las 2,700 imágenes para determinar:
1. Distribución de tipos de daño (labels 1-8)
2. Distribución de zonas del vehículo (zonas 1-10)
3. Densidad de defectos por imagen
4. Correlaciones zona-tipo de daño
5. Recomendaciones de estratificación para train/test

Output: Reporte detallado + visualizaciones
"""

from pathlib import Path
import json
from collections import Counter, defaultdict
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
sns.set_style("whitegrid")

# Mapeo de labels
DAMAGE_TYPES = {
    "1": "surface_scratch",
    "2": "dent",
    "3": "paint_peeling",
    "4": "deep_scratch",
    "5": "crack",
    "6": "missing_part",
    "7": "missing_accessory",
    "8": "misaligned_part"
}

# Mapeo de zonas (basado en la imagen)
VEHICLE_ZONES = {
    "1": "front_left_fender",      # Guardabarros delantero izquierdo
    "2": "hood_center",             # Capó centro
    "3": "front_right_fender",      # Guardabarros delantero derecho
    "4": "rear_left_quarter",       # Aleta trasera izquierda
    "5": "rear_bumper",             # Paragolpes trasero
    "6": "rear_right_quarter",      # Aleta trasera derecha
    "7": "driver_side_door",        # Puerta lateral izquierda
    "8": "driver_side_rocker",      # Bajo puerta izquierda
    "9": "passenger_side_door",     # Puerta lateral derecha
    "10": "passenger_side_rocker"   # Bajo puerta derecha
}


class DatasetAnalyzer:
    """Analizador exhaustivo del dataset completo"""
    
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.json_files = list(dataset_dir.glob("*labelDANO_modificado.json"))
        
        print(f"{'='*70}")
        print(f"📊 ANALIZANDO DATASET COMPLETO")
        print(f"{'='*70}")
        print(f"Directorio: {dataset_dir}")
        print(f"JSONs encontrados: {len(self.json_files)}")
        print(f"{'='*70}\n")
    
    def analyze(self) -> Dict:
        """Ejecuta análisis completo"""
        
        results = {
            'total_images': 0,
            'total_defects': 0,
            'damage_type_distribution': Counter(),
            'zone_distribution': Counter(),
            'defects_per_image': [],
            'zone_damage_correlation': defaultdict(Counter),
            'images_by_density': {'low': [], 'medium': [], 'high': []},
            'image_sizes': [],
            'invalid_images': []
        }
        
        print("🔍 Procesando imágenes...\n")
        
        for json_path in self.json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                
                # Extraer zona del nombre de archivo
                zone = self._extract_zone(data['imagePath'])
                
                # Contar defectos por tipo
                defects = [shape for shape in data['shapes'] if shape['shape_type'] == 'polygon']
                num_defects = len(defects)
                
                if num_defects == 0:
                    continue
                
                results['total_images'] += 1
                results['total_defects'] += num_defects
                results['defects_per_image'].append(num_defects)
                results['zone_distribution'][zone] += 1
                results['image_sizes'].append((data['imageWidth'], data['imageHeight']))
                
                # Distribución de tipos de daño
                for defect in defects:
                    label = defect['label']
                    results['damage_type_distribution'][label] += 1
                    results['zone_damage_correlation'][zone][label] += 1
                
                # Clasificar por densidad
                if num_defects <= 5:
                    results['images_by_density']['low'].append(json_path.stem)
                elif num_defects <= 15:
                    results['images_by_density']['medium'].append(json_path.stem)
                else:
                    results['images_by_density']['high'].append(json_path.stem)
            
            except Exception as e:
                results['invalid_images'].append((json_path.name, str(e)))
        
        return results
    
    def _extract_zone(self, image_path: str) -> str:
        """Extrae número de zona del nombre de archivo"""
        # Formato: zona1_ko_2_3_1554114337244_zona_5_imageDANO_original.jpg
        parts = image_path.split('_')
        for i, part in enumerate(parts):
            if part == 'zona' and i + 1 < len(parts):
                zone_num = parts[i + 1]
                if zone_num.isdigit():
                    return zone_num
        return "unknown"
    
    def generate_report(self, results: Dict, output_dir: Path):
        """Genera reporte completo con visualizaciones"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"📊 RESULTADOS DEL ANÁLISIS")
        print(f"{'='*70}\n")
        
        # ============================================
        # 1. ESTADÍSTICAS GENERALES
        # ============================================
        print("## 1. ESTADÍSTICAS GENERALES\n")
        print(f"Total imágenes válidas:  {results['total_images']}")
        print(f"Total defectos:          {results['total_defects']}")
        print(f"Promedio defectos/img:   {np.mean(results['defects_per_image']):.2f}")
        print(f"Mediana defectos/img:    {np.median(results['defects_per_image']):.0f}")
        print(f"Std defectos/img:        {np.std(results['defects_per_image']):.2f}")
        print(f"Min defectos:            {min(results['defects_per_image'])}")
        print(f"Max defectos:            {max(results['defects_per_image'])}")
        
        if results['invalid_images']:
            print(f"\n⚠️  Imágenes inválidas:    {len(results['invalid_images'])}")
        
        # ============================================
        # 2. DISTRIBUCIÓN POR TIPO DE DAÑO
        # ============================================
        print(f"\n{'='*70}")
        print("## 2. DISTRIBUCIÓN POR TIPO DE DAÑO\n")
        
        total_defects = sum(results['damage_type_distribution'].values())
        
        print(f"{'Label':<8} {'Tipo':<22} {'Cantidad':>10} {'%':>8}")
        print("-" * 50)
        
        sorted_types = sorted(
            results['damage_type_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for label, count in sorted_types:
            damage_name = DAMAGE_TYPES.get(label, "unknown")
            percentage = (count / total_defects) * 100
            print(f"{label:<8} {damage_name:<22} {count:>10} {percentage:>7.2f}%")
        
        # Calcular desbalance
        max_count = sorted_types[0][1]
        min_count = sorted_types[-1][1]
        imbalance_ratio = max_count / min_count
        
        print(f"\n⚠️  Ratio de desbalance: {imbalance_ratio:.1f}x")
        if imbalance_ratio > 50:
            print("   → DESBALANCE EXTREMO (considerar oversampling)")
        elif imbalance_ratio > 10:
            print("   → DESBALANCE ALTO (considerar ponderación en pérdida)")
        
        # ============================================
        # 3. DISTRIBUCIÓN POR ZONA
        # ============================================
        print(f"\n{'='*70}")
        print("## 3. DISTRIBUCIÓN POR ZONA DEL VEHÍCULO\n")
        
        print(f"{'Zona':<6} {'Descripción':<30} {'Imágenes':>10}")
        print("-" * 50)
        
        sorted_zones = sorted(
            results['zone_distribution'].items(),
            key=lambda x: int(x[0]) if x[0].isdigit() else 999
        )
        
        for zone, count in sorted_zones:
            zone_name = VEHICLE_ZONES.get(zone, "unknown")
            print(f"{zone:<6} {zone_name:<30} {count:>10}")
        
        # ============================================
        # 4. CORRELACIÓN ZONA-DAÑO
        # ============================================
        print(f"\n{'='*70}")
        print("## 4. CORRELACIÓN ZONA-TIPO DE DAÑO (Top 3 por zona)\n")
        
        for zone, damage_counts in sorted(results['zone_damage_correlation'].items()):
            zone_name = VEHICLE_ZONES.get(zone, "unknown")
            top3 = damage_counts.most_common(3)
            
            print(f"Zona {zone} ({zone_name}):")
            for label, count in top3:
                damage_name = DAMAGE_TYPES.get(label, "unknown")
                print(f"  - {damage_name}: {count}")
            print()
        
        # ============================================
        # 5. DISTRIBUCIÓN POR DENSIDAD
        # ============================================
        print(f"{'='*70}")
        print("## 5. DISTRIBUCIÓN POR DENSIDAD DE DEFECTOS\n")
        
        for density, images in results['images_by_density'].items():
            percentage = (len(images) / results['total_images']) * 100
            print(f"{density.upper():<10} ({percentage:>5.1f}%): {len(images):>5} imágenes")
        
        # ============================================
        # 6. VISUALIZACIONES
        # ============================================
        self._generate_visualizations(results, output_dir)
        
        # ============================================
        # 7. RECOMENDACIONES
        # ============================================
        print(f"\n{'='*70}")
        print("## 6. RECOMENDACIONES PARA TRAIN/TEST SPLIT\n")
        
        self._generate_recommendations(results)
        
        # Guardar resultados completos
        report_path = output_dir / "dataset_analysis_report.json"
        
        # Convertir Counter a dict para serialización
        serializable_results = {
            'total_images': results['total_images'],
            'total_defects': results['total_defects'],
            'damage_type_distribution': dict(results['damage_type_distribution']),
            'zone_distribution': dict(results['zone_distribution']),
            'defects_per_image_stats': {
                'mean': float(np.mean(results['defects_per_image'])),
                'median': float(np.median(results['defects_per_image'])),
                'std': float(np.std(results['defects_per_image'])),
                'min': int(min(results['defects_per_image'])),
                'max': int(max(results['defects_per_image']))
            },
            'density_distribution': {
                k: len(v) for k, v in results['images_by_density'].items()
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Reporte completo guardado en: {report_path}")
        print(f"📊 Visualizaciones guardadas en: {output_dir}")
        print(f"\n{'='*70}\n")
    
    def _generate_visualizations(self, results: Dict, output_dir: Path):
        """Genera visualizaciones del análisis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Distribución de tipos de daño
        labels = [DAMAGE_TYPES.get(k, k) for k in results['damage_type_distribution'].keys()]
        counts = list(results['damage_type_distribution'].values())
        
        axes[0, 0].bar(range(len(labels)), counts, color='steelblue')
        axes[0, 0].set_xticks(range(len(labels)))
        axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 0].set_title('Distribución de Tipos de Daño', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Cantidad')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Distribución de zonas
        zones = [VEHICLE_ZONES.get(k, k) for k in results['zone_distribution'].keys()]
        zone_counts = list(results['zone_distribution'].values())
        
        axes[0, 1].barh(range(len(zones)), zone_counts, color='coral')
        axes[0, 1].set_yticks(range(len(zones)))
        axes[0, 1].set_yticklabels(zones)
        axes[0, 1].set_title('Distribución de Zonas del Vehículo', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Cantidad de Imágenes')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Histograma de defectos por imagen
        axes[1, 0].hist(results['defects_per_image'], bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(results['defects_per_image']), color='red', linestyle='--', label=f"Media: {np.mean(results['defects_per_image']):.1f}")
        axes[1, 0].axvline(np.median(results['defects_per_image']), color='blue', linestyle='--', label=f"Mediana: {np.median(results['defects_per_image']):.0f}")
        axes[1, 0].set_title('Distribución de Defectos por Imagen', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Número de Defectos')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Densidad: Low/Medium/High
        density_labels = ['Low\n(≤5)', 'Medium\n(6-15)', 'High\n(>15)']
        density_counts = [
            len(results['images_by_density']['low']),
            len(results['images_by_density']['medium']),
            len(results['images_by_density']['high'])
        ]
        
        colors = ['lightgreen', 'gold', 'tomato']
        axes[1, 1].pie(density_counts, labels=density_labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1, 1].set_title('Distribución por Densidad de Defectos', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_analysis_plots.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Gráficos guardados: dataset_analysis_plots.png")
        
        plt.close()
    
    def _generate_recommendations(self, results: Dict):
        """Genera recomendaciones para train/test split"""
        
        total_images = results['total_images']
        
        # Calcular proporciones recomendadas
        recommended_train = int(total_images * 0.80)
        recommended_test = int(total_images * 0.20)
        
        print(f"📋 Configuración Recomendada:\n")
        print(f"1. TRAIN SET: {recommended_train} imágenes (80%)")
        print(f"   - Suficiente para índice FAISS robusto")
        print(f"   - ~{results['total_defects'] * 0.80:.0f} defectos para entrenamiento")
        
        print(f"\n2. TEST SET: {recommended_test} imágenes (20%)")
        print(f"   - Evaluación confiable con intervalo de confianza <5%")
        
        print(f"\n3. ESTRATIFICACIÓN:")
        print(f"   ✅ Balancear por DENSIDAD (low/medium/high)")
        print(f"   ✅ Balancear por ZONA (distribuir 10 zonas proporcionalmente)")
        print(f"   ⚠️  NO balancear por tipo (mantener distribución natural)")
        
        # Advertencias
        imbalance_ratio = max(results['damage_type_distribution'].values()) / min(results['damage_type_distribution'].values())
        
        if imbalance_ratio > 50:
            print(f"\n⚠️  ADVERTENCIA: Desbalance extremo ({imbalance_ratio:.0f}x)")
            print(f"   Considerar:")
            print(f"   - Data augmentation para clases minoritarias")
            print(f"   - Focal loss o class weighting")
            print(f"   - Oversampling de clases raras")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Análisis exhaustivo del dataset completo'
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path("/home/carlos/Escritorio/Proyectos-Minsait/Mapfre/carpetas-datos/jsons_segmentacion_jsonsfinales"),
        help='Directorio con JSONs del dataset'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("outputs/dataset_analysis"),
        help='Directorio de salida'
    )
    
    args = parser.parse_args()
    
    # Analizar
    analyzer = DatasetAnalyzer(args.dataset_dir)
    results = analyzer.analyze()
    analyzer.generate_report(results, args.output)


if __name__ == "__main__":
    main()
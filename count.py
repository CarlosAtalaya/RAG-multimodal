# count_defects.py

import json
from pathlib import Path
from collections import Counter

def count_defects_in_metadata(metadata_path: Path):
    """
    Analiza archivo de metadata y cuenta defectos
    
    Args:
        metadata_path: Ruta al archivo JSON de metadata
    """
    
    print(f"\n{'='*70}")
    print(f"📊 ANÁLISIS DE DEFECTOS EN METADATA")
    print(f"{'='*70}\n")
    
    # Cargar metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"📄 Archivo: {metadata_path.name}")
    print(f"📦 Total clusters: {len(metadata)}\n")
    
    # Contadores
    total_defects = 0
    defects_by_type = Counter()
    clusters_by_type = Counter()
    
    single_defect_clusters = 0
    multi_defect_clusters = 0
    
    defects_per_cluster = []
    
    # Analizar cada cluster
    for cluster in metadata:
        # Contar defectos en el cluster
        defect_count = cluster['defect_count']
        total_defects += defect_count
        defects_per_cluster.append(defect_count)
        
        # Tipo de cluster
        if defect_count == 1:
            single_defect_clusters += 1
        else:
            multi_defect_clusters += 1
        
        # Contar por tipo de daño
        dominant_type = cluster['dominant_type']
        clusters_by_type[dominant_type] += 1
        
        # Contar todos los tipos en el cluster
        for damage_type in cluster['damage_types']:
            defects_by_type[damage_type] += 1
    
    # ========================================
    # RESULTADOS
    # ========================================
    
    print(f"{'='*70}")
    print(f"✅ RESULTADOS")
    print(f"{'='*70}\n")
    
    print(f"🔢 Total defectos: {total_defects}")
    print(f"📦 Total clusters: {len(metadata)}")
    print(f"📉 Reducción: {(1 - len(metadata)/total_defects)*100:.1f}%")
    print(f"📊 Ratio: {total_defects/len(metadata):.2f} defectos/cluster\n")
    
    # Distribución de clusters
    print(f"{'─'*70}")
    print(f"📦 DISTRIBUCIÓN DE CLUSTERS")
    print(f"{'─'*70}\n")
    
    print(f"Single-defect clusters: {single_defect_clusters} ({single_defect_clusters/len(metadata)*100:.1f}%)")
    print(f"Multi-defect clusters:  {multi_defect_clusters} ({multi_defect_clusters/len(metadata)*100:.1f}%)\n")
    
    if multi_defect_clusters > 0:
        multi_defects = [d for d in defects_per_cluster if d > 1]
        print(f"Promedio defectos en multi-clusters: {sum(multi_defects)/len(multi_defects):.2f}")
        print(f"Máximo defectos en un cluster: {max(defects_per_cluster)}\n")
    
    # Distribución por tipo de daño
    print(f"{'─'*70}")
    print(f"🏷️  DISTRIBUCIÓN POR TIPO DE DAÑO")
    print(f"{'─'*70}\n")
    
    print("Defectos totales por tipo:")
    for damage_type, count in sorted(defects_by_type.items(), key=lambda x: -x[1]):
        percentage = (count / total_defects * 100)
        print(f"  {damage_type:20} : {count:4} ({percentage:5.1f}%)")
    
    print(f"\nClusters por tipo dominante:")
    for damage_type, count in sorted(clusters_by_type.items(), key=lambda x: -x[1]):
        percentage = (count / len(metadata) * 100)
        print(f"  {damage_type:20} : {count:4} ({percentage:5.1f}%)")
    
    print(f"\n{'='*70}\n")
    
    # Retornar estadísticas
    return {
        'total_defects': total_defects,
        'total_clusters': len(metadata),
        'reduction_percent': (1 - len(metadata)/total_defects)*100,
        'single_defect_clusters': single_defect_clusters,
        'multi_defect_clusters': multi_defect_clusters,
        'defects_by_type': dict(defects_by_type),
        'clusters_by_type': dict(clusters_by_type),
        'max_defects_per_cluster': max(defects_per_cluster),
        'avg_defects_per_cluster': total_defects / len(metadata)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analiza metadata de crops clusterizados y cuenta defectos'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path("data/processed/crops/train_815_originalimages/metadata/clustered_crops_metadata.json"),
        help='Ruta al archivo JSON de metadata'
    )
    parser.add_argument(
        '--export-stats',
        type=Path,
        default=None,
        help='Exportar estadísticas a JSON (opcional)'
    )
    
    args = parser.parse_args()
    
    # Verificar que existe el archivo
    if not args.metadata.exists():
        print(f"❌ Error: Archivo no encontrado")
        print(f"   Ruta: {args.metadata}")
        exit(1)
    
    # Analizar metadata
    stats = count_defects_in_metadata(args.metadata)
    
    # Exportar estadísticas si se solicita
    if args.export_stats:
        args.export_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_stats, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"💾 Estadísticas exportadas: {args.export_stats}\n")
# scripts/01_prepare_poc_dataset.py

import json
import random
from pathlib import Path
from collections import defaultdict, Counter
import shutil

def select_balanced_poc_dataset(
    source_dir: Path,
    output_dir: Path,
    target_samples: int = 100,
    min_damages_per_image: int = 5
):
    """
    Selecciona 100 imágenes balanceadas por:
    1. Diversidad de tipos de daño
    2. Número de defectos (evitar imágenes con 1 solo defecto)
    3. Zonas del vehículo
    """
    
    # Escanear todos los JSONs
    json_files = list(source_dir.glob("*labelDANO_modificado.json"))
    print(f"📊 Total de imágenes disponibles: {len(json_files)}")
    
    # Analizar dataset
    damage_stats = defaultdict(list)
    image_metadata = []
    
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Contar daños por tipo
        damage_counts = Counter([
            shape['label'] for shape in data['shapes']
        ])
        
        # Solo considerar imágenes con suficientes defectos
        total_damages = sum(damage_counts.values())
        if total_damages < min_damages_per_image:
            continue
        
        # Extraer zona del vehículo del nombre
        image_name = data['imagePath']
        zone = extract_vehicle_zone(image_name)
        
        metadata = {
            'json_path': json_path,
            'image_path': source_dir / image_name,
            'zone': zone,
            'total_damages': total_damages,
            'damage_distribution': dict(damage_counts),
            'dominant_type': damage_counts.most_common(1)[0][0]
        }
        
        image_metadata.append(metadata)
        
        # Agrupar por tipo dominante
        for damage_type in damage_counts.keys():
            damage_stats[damage_type].append(metadata)
    
    print(f"✅ Imágenes con ≥{min_damages_per_image} defectos: {len(image_metadata)}")
    
    # Estrategia de muestreo balanceado
    samples_per_type = target_samples // 8  # 8 tipos de daño
    
    selected = []
    for damage_type in range(1, 9):
        str_type = str(damage_type)
        available = damage_stats[str_type]
        
        if not available:
            print(f"⚠️  Tipo {str_type}: No hay imágenes suficientes")
            continue
        
        # Seleccionar aleatoriamente manteniendo diversidad
        sampled = random.sample(
            available,
            min(samples_per_type, len(available))
        )
        selected.extend(sampled)
        
        print(f"✓ Tipo {str_type}: {len(sampled)} imágenes seleccionadas")
    
    # Si no llegamos a 100, completar con imágenes aleatorias
    if len(selected) < target_samples:
        remaining = [m for m in image_metadata if m not in selected]
        additional = random.sample(
            remaining,
            min(target_samples - len(selected), len(remaining))
        )
        selected.extend(additional)
    
    # Limitar a target_samples
    selected = selected[:target_samples]
    
    print(f"\n📦 DATASET POC FINAL: {len(selected)} imágenes")
    print_dataset_stats(selected)
    
    # Copiar archivos seleccionados
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    for idx, meta in enumerate(selected):
        # Copiar imagen
        img_dst = output_dir / meta['image_path'].name
        shutil.copy2(meta['image_path'], img_dst)
        
        # Copiar JSON
        json_dst = output_dir / meta['json_path'].name
        shutil.copy2(meta['json_path'], json_dst)
        
        manifest.append({
            'id': idx,
            'image': meta['image_path'].name,
            'json': meta['json_path'].name,
            'zone': meta['zone'],
            'total_damages': meta['total_damages'],
            'damage_distribution': meta['damage_distribution']
        })
    
    # Guardar manifest
    manifest_path = output_dir / 'poc_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Dataset POC guardado en: {output_dir}")
    print(f"📄 Manifest: {manifest_path}")
    
    return manifest

def extract_vehicle_zone(image_name: str) -> str:
    """Extrae zona del vehículo del nombre del archivo"""
    # Formato: zona1_ko_2_3_1554114337244_zona_5_imageDANO_original.jpg
    parts = image_name.split('_')
    for i, part in enumerate(parts):
        if part == 'zona' and i + 1 < len(parts):
            return f"zone_{parts[i+1]}"
    return "unknown"

def print_dataset_stats(selected):
    """Imprime estadísticas del dataset seleccionado"""
    zones = Counter([m['zone'] for m in selected])
    total_damages = sum([m['total_damages'] for m in selected])
    
    print("\n📊 ESTADÍSTICAS:")
    print(f"  - Total imágenes: {len(selected)}")
    print(f"  - Total defectos: {total_damages}")
    print(f"  - Promedio defectos/imagen: {total_damages/len(selected):.1f}")
    print(f"\n  Distribución por zona:")
    for zone, count in zones.most_common():
        print(f"    {zone}: {count} imágenes")

if __name__ == "__main__":
    SOURCE_DIR = Path("/home/carlos/Escritorio/Proyectos-Minsait/Mapfre/carpetas-datos/jsons_segmentacion_jsonsfinales")
    OUTPUT_DIR = Path("/home/carlos/Escritorio/Proyectos-Minsait/Mapfre/RAG-multimodal/100_samples/")
    
    manifest = select_balanced_poc_dataset(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        target_samples=100
    )
    
    print("\n✨ Preparación completada!")
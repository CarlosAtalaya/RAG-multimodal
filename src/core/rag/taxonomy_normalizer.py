# src/core/rag/taxonomy_normalizer.py

from typing import Dict, List

class TaxonomyNormalizer:
    """
    Normaliza labels de entrenamiento a taxonomía de benchmark
    
    Mapeo completo: train labels (1-8, surface_scratch, etc.) 
                  → benchmark labels (Scratch, Dent, etc.)
    """
    
    # Taxonomía oficial del benchmark
    BENCHMARK_DAMAGES = [
        "Scratch", "Dent", "Degraded varnish", "Crack",
        "Fractured part", "Missing part", "Deviated part",
        "No damage", "Unknown"
    ]
    
    # Mapeo semántico: train → benchmark
    MAPPING = {
        # Labels numéricos (dataset original)
        "1": ("Scratch", 1.0),
        "2": ("Dent", 1.0),
        "3": ("Degraded varnish", 1.0),
        "4": ("Scratch", 1.0),
        "5": ("Crack", 1.0),
        "6": ("Missing part", 1.0),
        "7": ("Missing part", 0.9),
        "8": ("Deviated part", 1.0),
        
        # Labels canónicos (crop_generator)
        "surface_scratch": ("Scratch", 1.0),
        "deep_scratch": ("Scratch", 1.0),
        "dent": ("Dent", 1.0),
        "paint_peeling": ("Degraded varnish", 1.0),
        "crack": ("Crack", 1.0),
        "missing_part": ("Missing part", 1.0),
        "missing_accessory": ("Missing part", 0.9),
        "misaligned_part": ("Deviated part", 1.0),
        
        # Casos especiales
        "unknown": ("Unknown", 1.0),
        "no_damage": ("No damage", 1.0)
    }
    
    def normalize(self, train_label: str) -> Dict:
        """
        Normaliza un label de train a benchmark
        
        Returns:
            {
                'benchmark_label': str,
                'confidence': float,
                'original': str
            }
        """
        train_label = train_label.lower().strip()
        
        if train_label not in self.MAPPING:
            return {
                'benchmark_label': 'Unknown',
                'confidence': 0.0,
                'original': train_label,
                'error': f'NO_MAPPING_FOR_{train_label}'
            }
        
        benchmark_label, confidence = self.MAPPING[train_label]
        
        return {
            'benchmark_label': benchmark_label,
            'confidence': confidence,
            'original': train_label
        }
    
    def normalize_batch(self, labels: List[str]) -> List[Dict]:
        """Normaliza lista de labels"""
        return [self.normalize(label) for label in labels]
    
    def get_coverage_stats(self, train_labels: List[str]) -> Dict:
        """Calcula estadísticas de cobertura"""
        from collections import Counter
        
        counts = Counter(train_labels)
        total = sum(counts.values())
        
        mapped = sum(c for label, c in counts.items() if label.lower().strip() in self.MAPPING)
        
        return {
            'total_samples': total,
            'mapped_samples': mapped,
            'unmapped_samples': total - mapped,
            'coverage_percent': (mapped / total * 100) if total > 0 else 0.0,
            'label_distribution': dict(counts)
        }


# Test rápido
if __name__ == "__main__":
    n = TaxonomyNormalizer()
    
    tests = ["surface_scratch", "paint_peeling", "1", "3", "unknown_label"]
    
    print("🧪 Taxonomy Normalizer Test\n")
    for label in tests:
        r = n.normalize(label)
        status = "✅" if r['confidence'] > 0 else "❌"
        print(f"{status} {label:20} → {r['benchmark_label']:20} ({r['confidence']:.0%})")
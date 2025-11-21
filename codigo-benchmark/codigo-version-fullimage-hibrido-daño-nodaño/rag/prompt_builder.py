# rag/prompt_builder.py

from typing import List, Dict


class RAGPromptBuilder:
    """
    Constructor de prompts con contexto RAG normalizado
    
    Compatible con:
    - Full images CON daño (con zona vehículo, descripciones)
    - Full images SIN daño (con zona vehículo, verificación de calidad)
    - Crops (legacy, con spatial_zone)
    """
    
    def inject_rag_context(
        self,
        original_prompt: str,
        search_results: list,
        max_examples: int = 3,
        balance: bool = False,
        min_damage_examples: int = 2
    ) -> str:
        """
        Inyecta contexto RAG en el prompt original
        
        Args:
            original_prompt: Prompt original de evaluación
            search_results: Lista de SearchResult del retriever
            max_examples: Número máximo de ejemplos a incluir
            balance: Si True, balancea ejemplos con/sin daño
            min_damage_examples: Mínimo de ejemplos CON daño (si balance=True)
        
        Returns:
            Prompt con contexto RAG inyectado
        """
        
        if not search_results:
            return original_prompt
        
        # ✨ Balancear ejemplos si está activado
        if balance:
            examples = self.balance_examples(
                search_results, 
                max_examples, 
                min_damage_examples
            )
        else:
            examples = search_results[:max_examples]
        
        # Construir contexto con labels NORMALIZADOS
        context_parts = [
            "\n## 🔍 Similar Verified Cases from Database:\n",
            "The following examples show similar patterns:\n"
        ]
        
        for i, result in enumerate(examples, 1):
            context_parts.append(f"\n### Example {i}:")
            
            # Descripción textual (PRIORIZAR)
            if result.description:
                context_parts.append(f"- **Description**: {result.description}")
            
            # ✨ NUEVO: Manejo diferenciado por has_damage
            if result.has_damage:
                # ===== CASO CON DAÑO =====
                
                # Tipos de daño (normalizados)
                if result.damage_types and len(result.damage_types) > 1:
                    types_str = ", ".join(set(result.damage_types))
                    context_parts.append(f"- **Damage types**: {types_str}")
                else:
                    context_parts.append(f"- **Damage type**: {result.damage_type}")
                
                # Total defectos
                if result.total_defects:
                    context_parts.append(f"- **Total defects**: {result.total_defects}")
                
                # Distribución de defectos (opcional)
                if result.defect_distribution:
                    dist_str = ", ".join(
                        f"{count}x {dtype}" 
                        for dtype, count in sorted(
                            result.defect_distribution.items(), 
                            key=lambda x: -x[1]
                        )
                    )
                    context_parts.append(f"- **Distribution**: {dist_str}")
                
            else:
                # ===== CASO SIN DAÑO =====
                context_parts.append(f"- **Damage status**: No visible damage")
                context_parts.append(f"- **Quality**: Clean surface verified")
            
            # Zona del vehículo (común para ambos)
            if result.zone_description != 'unknown':
                context_parts.append(
                    f"- **Vehicle zone**: {result.zone_description} ({result.zone_area})"
                )
            # Zona espacial (legacy - crops)
            elif result.spatial_zone != 'unknown':
                context_parts.append(
                    f"- **Vehicle area**: {self._format_zone(result.spatial_zone)}"
                )
            
            # Similitud visual (común)
            similarity = (1 - result.distance) * 100
            context_parts.append(f"- **Visual similarity**: {similarity:.1f}%")
        
        # ✨ Resumen de ejemplos
        n_damage = sum(1 for r in examples if r.has_damage)
        n_no_damage = len(examples) - n_damage
        
        context_parts.append("\n---\n")
        context_parts.append(
            f"**Summary**: {len(examples)} verified examples "
            f"({n_damage} with damage, {n_no_damage} without damage)\n"
        )
        
        context_parts.append("\n## 📋 Your Task:\n")
        context_parts.append(original_prompt)
        
        return "\n".join(context_parts)
    
    def balance_examples(
        self, 
        search_results: List, 
        max_examples: int = 3,
        min_damage_examples: int = 2
    ) -> List:
        """
        Balancea ejemplos para incluir tanto con daño como sin daño
        
        Args:
            search_results: Resultados de búsqueda (SearchResult)
            max_examples: Total de ejemplos
            min_damage_examples: Mínimo de ejemplos CON daño
        
        Returns:
            Lista balanceada de SearchResult
        """
        damage_results = [r for r in search_results if r.has_damage]
        no_damage_results = [r for r in search_results if not r.has_damage]
        
        balanced = []
        
        # 1. Priorizar ejemplos con daño (si existen)
        if damage_results:
            balanced.extend(damage_results[:min_damage_examples])
        
        # 2. Rellenar con sin daño si hay espacio
        remaining_slots = max_examples - len(balanced)
        if remaining_slots > 0 and no_damage_results:
            balanced.extend(no_damage_results[:remaining_slots])
        
        # 3. Si no llegamos a max_examples, añadir más con daño
        if len(balanced) < max_examples and damage_results:
            additional_needed = max_examples - len(balanced)
            start_idx = min(min_damage_examples, len(damage_results))
            balanced.extend(damage_results[start_idx:start_idx + additional_needed])
        
        return balanced[:max_examples]
    
    def _format_zone(self, spatial_zone: str) -> str:
        """Traduce zonas espaciales (legacy crops)"""
        zone_map = {
            "top_left": "Upper left area",
            "top_center": "Upper center",
            "top_right": "Upper right area",
            "middle_left": "Left side",
            "middle_center": "Center",
            "middle_right": "Right side",
            "bottom_left": "Lower left area",
            "bottom_center": "Lower center",
            "bottom_right": "Lower right area"
        }
        return zone_map.get(spatial_zone, spatial_zone)
    
    def inject_rag_context_simple(
        self,
        original_prompt: str,
        search_results: list,
        max_examples: int = 3
    ) -> str:
        """
        Versión simplificada sin balanceo (compatibilidad)
        """
        return self.inject_rag_context(
            original_prompt=original_prompt,
            search_results=search_results,
            max_examples=max_examples,
            balance=False
        )
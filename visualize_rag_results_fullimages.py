# visualize_rag_results_fullimages.py (VERSIÓN CORREGIDA)

import streamlit as st
import json
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="RAG Hybrid Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #3b82f6;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .image-card {
        border: 2px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .ground-truth {
        background-color: #dcfce7;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
    }
    .generated-answer {
        background-color: #fef3c7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
        max-height: 400px;
        overflow-y: auto;
    }
    .zone-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.25rem;
    }
    .zone-frontal { background-color: #dbeafe; color: #1e40af; }
    .zone-posterior { background-color: #fce7f3; color: #9f1239; }
    .zone-lateral-left { background-color: #dcfce7; color: #166534; }
    .zone-lateral-right { background-color: #fef3c7; color: #92400e; }
    .zone-unknown { background-color: #f3f4f6; color: #374151; }
</style>
""", unsafe_allow_html=True)

def load_results(json_path: Path):
    """Carga resultados de evaluación"""
    with open(json_path) as f:
        data = json.load(f)
    return data

def get_damage_name(label: str) -> str:
    """Convierte label a nombre legible"""
    mapping = {
        "1": "Surface Scratch",
        "2": "Dent",
        "3": "Paint Peeling",
        "4": "Deep Scratch",
        "5": "Crack",
        "6": "Missing Part",
        "7": "Missing Accessory",
        "8": "Misaligned Part",
        "surface_scratch": "Surface Scratch",
        "deep_scratch": "Deep Scratch",
        "dent": "Dent",
        "paint_peeling": "Paint Peeling",
        "crack": "Crack",
        "missing_part": "Missing Part",
        "missing_accessory": "Missing Accessory",
        "misaligned_part": "Misaligned Part",
        # Benchmark labels
        "Scratch": "Scratch",
        "Dent": "Dent",
        "Degraded varnish": "Degraded Varnish",
        "Crack": "Crack",
        "Missing part": "Missing Part",
        "Deviated part": "Deviated Part"
    }
    return mapping.get(label, label.replace('_', ' ').title())

def get_zone_badge_html(zone_area: str) -> str:
    """Genera badge HTML para zona"""
    zone_class = f"zone-{zone_area.replace('_', '-')}"
    return f'<span class="zone-badge {zone_class}">{zone_area.replace("_", " ").title()}</span>'

def display_query_image(image_path: str, ground_truth: dict, case_id: int):
    """Muestra imagen query con ground truth"""
    st.markdown(f'<div class="sub-header">🖼️ Query Image #{case_id}</div>', unsafe_allow_html=True)
    
    # Ground truth
    st.markdown('<div class="ground-truth">', unsafe_allow_html=True)
    st.markdown("**Ground Truth:**")
    
    for label, count in ground_truth.items():
        damage_name = get_damage_name(label)
        st.markdown(f"- **{damage_name}**: {count} defect(s)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Imagen
    try:
        img = Image.open(image_path)
        st.image(img, use_container_width=True, caption=Path(image_path).name)
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")

def display_retrieved_images(retrieved: list):
    """Muestra imágenes retrieved en grid"""
    st.markdown('<div class="sub-header">🔍 Top-5 Retrieved Similar Images</div>', unsafe_allow_html=True)
    
    if not retrieved:
        st.warning("No retrieved images available")
        return
    
    # Grid de 5 columnas
    cols = st.columns(5)
    
    for idx, img_info in enumerate(retrieved):
        with cols[idx]:
            img_path = img_info.get('image_path', '')
            
            if not img_path or not Path(img_path).exists():
                st.warning(f"Image {idx+1} not found")
                continue
            
            try:
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
                
                # Información
                similarity = (1 - img_info.get('distance', 1.0)) * 100
                
                # Zona con badge
                zone_area = img_info.get('zone_area', 'unknown')
                zone_badge = get_zone_badge_html(zone_area)
                
                # Tipos de daño (limitar a 3)
                damage_types = img_info.get('damage_types', [])
                if damage_types:
                    damage_types_html = "<br>".join([
                        f"• {get_damage_name(dt)}" 
                        for dt in damage_types[:3]
                    ])
                else:
                    damage_types_html = "• No damage info"
                
                st.markdown(f"""
                <div class="image-card">
                    <b>#{idx + 1}</b><br>
                    {zone_badge}<br>
                    <b>Zone:</b> {img_info.get('vehicle_zone', 'N/A')}<br>
                    <b>Damages:</b><br>
                    {damage_types_html}<br>
                    <b>Similarity:</b> {similarity:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error loading image {idx+1}: {e}")

def display_generated_answer(answer: str, recall: float, times: dict):
    """Muestra respuesta generada y métricas"""
    st.markdown('<div class="sub-header">🤖 RAG Generated Response</div>', unsafe_allow_html=True)
    
    # Métricas en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        recall_color = "🟢" if recall > 0.5 else "🟡" if recall > 0.2 else "🔴"
        st.metric("Recall@5", f"{recall_color} {recall:.1%}")
    with col2:
        st.metric("Retrieval", f"{times.get('retrieval_time_ms', 0):.0f}ms")
    with col3:
        st.metric("Generation", f"{times.get('generation_time_ms', 0)/1000:.1f}s")
    with col4:
        total_time = times.get('retrieval_time_ms', 0) + times.get('generation_time_ms', 0)
        st.metric("Total", f"{total_time/1000:.2f}s")
    
    # Respuesta generada
    st.markdown('<div class="generated-answer">', unsafe_allow_html=True)
    st.markdown("**Generated Analysis:**")
    
    # Intentar parsear JSON
    try:
        answer_json = json.loads(answer.strip().strip('```json').strip('```'))
        
        if 'damages' in answer_json:
            st.markdown("**Detected Damages:**")
            for damage in answer_json['damages']:
                st.markdown(
                    f"- **{damage.get('type', 'Unknown')}** "
                    f"at *{damage.get('location', 'N/A')}* "
                    f"(Severity: {damage.get('severity', 'N/A')})"
                )
        
        if 'summary' in answer_json:
            st.markdown(f"\n**Summary:** {answer_json['summary']}")
    except:
        # Si no es JSON, mostrar como texto
        st.write(answer)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">📊 RAG Hybrid Embeddings Visualizer</div>', unsafe_allow_html=True)
    
    # ============================================
    # SIDEBAR - CONTROLES
    # ============================================
    
    st.sidebar.title("⚙️ Configuration")
    
    # Path al JSON
    json_path = st.sidebar.text_input(
        "Results JSON Path",
        value="outputs/rag_evaluation_hybrid/evaluation_results_hybrid.json"
    )
    
    if not Path(json_path).exists():
        st.error(f"❌ File not found: {json_path}")
        st.stop()
    
    # Cargar resultados
    try:
        data = load_results(Path(json_path))
    except Exception as e:
        st.error(f"❌ Error loading JSON: {e}")
        st.stop()
    
    test_cases = data.get('test_cases', [])
    metrics = data.get('metrics', {})
    config = data.get('config', {})
    errors = data.get('errors', [])
    
    if not test_cases:
        st.warning("⚠️ No test cases found in results")
        st.stop()
    
    # ============================================
    # SIDEBAR - MÉTRICAS GLOBALES
    # ============================================
    
    st.sidebar.markdown("### 📈 Global Metrics")
    
    # Compatibilidad con ambos formatos de JSON
    total_cases = metrics.get('total_cases', metrics.get('num_cases', 0))
    successful_cases = metrics.get('successful_cases', len(test_cases))
    failed_cases = metrics.get('failed_cases', 0)
    
    st.sidebar.metric("Total Cases", total_cases)
    st.sidebar.metric("Successful", f"✅ {successful_cases}")
    
    if failed_cases > 0:
        st.sidebar.metric("Failed", f"❌ {failed_cases}")
    
    # Recall
    avg_recall = metrics.get('avg_recall_at_k', 0.0)
    recall_status = "🟢" if avg_recall > 0.5 else "🟡" if avg_recall > 0.2 else "🔴"
    
    st.sidebar.metric("Avg Recall@5", f"{recall_status} {avg_recall:.1%}")
    
    # Tiempos
    st.sidebar.metric("Avg Retrieval", f"{metrics.get('avg_retrieval_time_ms', 0):.0f}ms")
    st.sidebar.metric("Avg Generation", f"{metrics.get('avg_generation_time_ms', 0)/1000:.1f}s")
    
    # Config del modelo
    if config:
        st.sidebar.markdown("### ⚙️ Model Config")
        st.sidebar.text(f"Visual weight: {config.get('visual_weight', 'N/A')}")
        st.sidebar.text(f"Text weight: {config.get('text_weight', 'N/A')}")
        st.sidebar.text(f"Top-k: {config.get('k', 'N/A')}")
    
    # Diagnóstico
    if avg_recall < 0.2:
        st.sidebar.warning("⚠️ Low recall detected")
    elif avg_recall > 0.5:
        st.sidebar.success("✅ Good performance!")
    
    # Errores
    if errors:
        st.sidebar.markdown("### ⚠️ Errors")
        st.sidebar.error(f"{len(errors)} cases failed")
        with st.sidebar.expander("View errors"):
            for err in errors[:5]:
                st.text(f"• {err.get('image', 'Unknown')}")
                st.caption(f"  {err.get('error', 'No error msg')[:50]}...")
    
    # ============================================
    # NAVEGACIÓN
    # ============================================
    
    if 'case_idx' not in st.session_state:
        st.session_state.case_idx = 0
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    
    # Botones
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    
    with col1:
        if st.button("◀ Prev", use_container_width=True):
            if st.session_state.case_idx > 0:
                st.session_state.case_idx -= 1
                st.rerun()
    
    with col2:
        new_idx = st.number_input(
            "Case #",
            min_value=0,
            max_value=len(test_cases) - 1,
            value=st.session_state.case_idx,
            step=1,
            label_visibility="collapsed"
        )
        if new_idx != st.session_state.case_idx:
            st.session_state.case_idx = new_idx
            st.rerun()
    
    with col3:
        if st.button("Next ▶", use_container_width=True):
            if st.session_state.case_idx < len(test_cases) - 1:
                st.session_state.case_idx += 1
                st.rerun()
    
    # Dropdown
    st.sidebar.markdown("**OR select:**")
    case_options = [
        f"Case #{i} - Recall: {tc.get('recall_at_k', 0):.0%}" 
        for i, tc in enumerate(test_cases)
    ]
    selected_option = st.sidebar.selectbox(
        "Jump to case",
        case_options,
        index=st.session_state.case_idx,
        label_visibility="collapsed"
    )
    
    dropdown_idx = case_options.index(selected_option)
    if dropdown_idx != st.session_state.case_idx:
        st.session_state.case_idx = dropdown_idx
        st.rerun()
    
    # Obtener caso actual
    case_idx = st.session_state.case_idx
    test_case = test_cases[case_idx]
    
    # Opciones de visualización
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Display Options")
    show_full_answer = st.sidebar.checkbox("Show Full Answer", value=True)
    show_image_paths = st.sidebar.checkbox("Show Image Paths", value=False)
    show_zone_info = st.sidebar.checkbox("Show Zone Details", value=True)
    
    # ============================================
    # LAYOUT PRINCIPAL
    # ============================================
    
    # Progreso
    progress_text = f"**Viewing Case {case_idx + 1} of {len(test_cases)}**"
    st.markdown(progress_text)
    st.progress((case_idx + 1) / len(test_cases))
    
    st.markdown("---")
    
    # Fila 1: Query Image + Info
    col_query, col_info = st.columns([2, 1])
    
    with col_query:
        display_query_image(
            test_case.get('image_path', ''),
            test_case.get('ground_truth', {}),
            case_idx
        )
    
    with col_info:
        st.markdown('<div class="sub-header">ℹ️ Case Info</div>', unsafe_allow_html=True)
        
        recall = test_case.get('recall_at_k', 0.0)
        recall_status = "🟢" if recall > 0.5 else "🟡" if recall > 0.2 else "🔴"
        
        st.markdown(f"""
        <div class="metric-card">
            <b>Image ID:</b> {test_case.get('image_id', 'N/A')}<br>
            <b>File:</b> {Path(test_case.get('image_path', '')).name}<br>
            <b>Recall@5:</b> {recall_status} {recall:.1%}<br>
            <b>Retrieval:</b> {test_case.get('retrieval_time_ms', 0):.0f}ms<br>
            <b>Generation:</b> {test_case.get('generation_time_ms', 0)/1000:.1f}s
        </div>
        """, unsafe_allow_html=True)
        
        # Zona info
        if show_zone_info:
            retrieved = test_case.get('retrieved_images', [])
            if retrieved:
                st.markdown("**Retrieved Zones:**")
                zones = [img.get('vehicle_zone', 'N/A') for img in retrieved]
                zone_counts = {}
                for z in zones:
                    zone_counts[z] = zone_counts.get(z, 0) + 1
                for zone, count in zone_counts.items():
                    st.markdown(f"- Zone {zone}: {count}x")
        
        # Paths
        if show_image_paths:
            st.markdown("**Image Paths:**")
            for i, img in enumerate(test_case.get('retrieved_images', []), 1):
                st.code(f"{i}. {img.get('image_path', 'N/A')}", language=None)
    
    st.markdown("---")
    
    # Fila 2: Retrieved Images
    display_retrieved_images(test_case.get('retrieved_images', []))
    
    st.markdown("---")
    
    # Fila 3: Generated Answer
    answer = test_case.get('generated_answer', '')
    if not show_full_answer and len(answer) > 500:
        answer = answer[:500] + "...\n\n*(Truncated)*"
    
    display_generated_answer(
        answer,
        test_case.get('recall_at_k', 0.0),
        {
            'retrieval_time_ms': test_case.get('retrieval_time_ms', 0),
            'generation_time_ms': test_case.get('generation_time_ms', 0)
        }
    )
    
    # Footer
    st.markdown("---")
    st.caption("💡 **Tip:** Use ◀ Prev / Next ▶ buttons to navigate")
    st.caption("🔧 RAG Hybrid Embeddings Visualizer | Streamlit")

if __name__ == "__main__":
    main()
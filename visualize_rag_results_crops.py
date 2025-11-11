# visualize_rag_results.py (VERSIÓN MEJORADA)

import streamlit as st
import json
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="RAG Multimodal Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (igual que antes)
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
    .crop-card {
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
    /* Estilos para botones de navegación */
    .nav-button {
        font-size: 1.2rem;
        padding: 0.5rem 1.5rem;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_results(json_path: Path):
    """Carga resultados de evaluación"""
    with open(json_path) as f:
        data = json.load(f)
    return data

def display_query_image(image_path: str, ground_truth: dict, case_id: int):
    """Muestra imagen query con ground truth"""
    st.markdown(f'<div class="sub-header">🖼️ Query Image #{case_id}</div>', unsafe_allow_html=True)
    
    # Ground truth
    st.markdown('<div class="ground-truth">', unsafe_allow_html=True)
    st.markdown("**Ground Truth:**")
    
    damage_mapping = {
        "1": "surface_scratch",
        "2": "dent",
        "3": "paint_peeling",
        "4": "deep_scratch",
        "5": "crack",
        "6": "missing_part",
        "7": "missing_accessory",
        "8": "misaligned_part"
    }
    
    for label, count in ground_truth.items():
        damage_name = damage_mapping.get(label, f"Unknown ({label})")
        st.markdown(f"- **{damage_name}**: {count} defect(s)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Imagen
    try:
        img = Image.open(image_path)
        st.image(img, use_container_width=True, caption=Path(image_path).name)
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")

def display_retrieved_crops(crops: list):
    """Muestra crops retrieved en grid"""
    st.markdown('<div class="sub-header">🔍 Top-5 Retrieved Crops</div>', unsafe_allow_html=True)
    
    # Grid de 5 columnas
    cols = st.columns(5)
    
    for idx, crop_info in enumerate(crops):
        with cols[idx]:
            # Cargar imagen del crop
            crop_path = crop_info['crop_path']
            try:
                crop_img = Image.open(crop_path)
                st.image(crop_img, use_container_width=True)
                
                # Información del crop
                similarity = (1 - crop_info['distance']) * 100
                
                st.markdown(f"""
                <div class="crop-card">
                    <b>#{idx + 1}</b><br>
                    <b>Type:</b> {crop_info['damage_type']}<br>
                    <b>Zone:</b> {crop_info['spatial_zone']}<br>
                    <b>Similarity:</b> {similarity:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {e}")

def display_generated_answer(answer: str, recall: float, times: dict):
    """Muestra respuesta generada y métricas"""
    st.markdown('<div class="sub-header">🤖 RAG Generated Response</div>', unsafe_allow_html=True)
    
    # Métricas en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recall@5", f"{recall:.1%}")
    with col2:
        st.metric("Retrieval Time", f"{times['retrieval_time_ms']:.0f}ms")
    with col3:
        st.metric("Generation Time", f"{times['generation_time_ms']/1000:.1f}s")
    with col4:
        st.metric("Total Time", f"{(times['retrieval_time_ms'] + times['generation_time_ms'])/1000:.2f}s")
    
    # Respuesta generada
    st.markdown('<div class="generated-answer">', unsafe_allow_html=True)
    st.markdown("**Generated Analysis:**")
    st.write(answer)
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">📊 RAG Multimodal Results Visualizer</div>', unsafe_allow_html=True)
    
    # ============================================
    # SIDEBAR - CONTROLES
    # ============================================
    
    st.sidebar.title("⚙️ Configuration")
    
    # Path al JSON
    json_path = st.sidebar.text_input(
        "Results JSON Path",
        value="outputs/rag_evaluation/rag_dinov3_train815/rag_evaluation_results.json"
    )
    
    if not Path(json_path).exists():
        st.error(f"❌ File not found: {json_path}")
        st.stop()
    
    # Cargar resultados
    data = load_results(Path(json_path))
    test_cases = data['test_cases']
    metrics = data['metrics']
    
    # ============================================
    # NAVEGACIÓN CON SESSION STATE
    # ============================================
    
    # Inicializar índice en session state
    if 'case_idx' not in st.session_state:
        st.session_state.case_idx = 0
    
    # Sidebar - Métricas globales
    st.sidebar.markdown("### 📈 Global Metrics")
    st.sidebar.metric("Total Cases", metrics['num_cases'])
    st.sidebar.metric("Avg Recall@5", f"{metrics['avg_recall_at_k']:.1%}")
    st.sidebar.metric("Avg Retrieval", f"{metrics['avg_retrieval_time_ms']:.0f}ms")
    st.sidebar.metric("Avg Generation", f"{metrics['avg_generation_time_ms']/1000:.1f}s")
    
    # ============================================
    # NAVEGACIÓN: BOTONES + SELECTOR
    # ============================================
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    
    # Botones de navegación
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    
    with col1:
        if st.button("◀ Prev", use_container_width=True):
            if st.session_state.case_idx > 0:
                st.session_state.case_idx -= 1
                st.rerun()
    
    with col2:
        # Selector numérico
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
    
    # Selector dropdown alternativo
    st.sidebar.markdown("**OR select from dropdown:**")
    case_options = [f"Case #{i} - {Path(tc['image_path']).name[:30]}..." for i, tc in enumerate(test_cases)]
    selected_option = st.sidebar.selectbox(
        "Jump to case",
        case_options,
        index=st.session_state.case_idx,
        label_visibility="collapsed"
    )
    
    # Actualizar índice si cambia el dropdown
    dropdown_idx = case_options.index(selected_option)
    if dropdown_idx != st.session_state.case_idx:
        st.session_state.case_idx = dropdown_idx
        st.rerun()
    
    # Obtener caso actual
    case_idx = st.session_state.case_idx
    test_case = test_cases[case_idx]
    
    # Filtros opcionales
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Display Options")
    show_full_answer = st.sidebar.checkbox("Show Full Answer", value=True)
    show_crop_paths = st.sidebar.checkbox("Show Crop Paths", value=False)
    
    # ============================================
    # LAYOUT PRINCIPAL
    # ============================================
    
    # Indicador de progreso
    progress_text = f"**Viewing Case {case_idx + 1} of {len(test_cases)}**"
    st.markdown(progress_text)
    st.progress((case_idx + 1) / len(test_cases))
    
    st.markdown("---")
    
    # Fila 1: Query Image + Info
    col_query, col_info = st.columns([2, 1])
    
    with col_query:
        display_query_image(
            test_case['image_path'],
            test_case['ground_truth'],
            case_idx
        )
    
    with col_info:
        st.markdown('<div class="sub-header">ℹ️ Case Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card">
            <b>Image ID:</b> {test_case['image_id']}<br>
            <b>File:</b> {Path(test_case['image_path']).name}<br>
            <b>Recall@5:</b> {test_case['recall_at_k']:.1%}<br>
            <b>Retrieval:</b> {test_case['retrieval_time_ms']:.0f}ms<br>
            <b>Generation:</b> {test_case['generation_time_ms']/1000:.1f}s
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar paths de crops si está activado
        if show_crop_paths:
            st.markdown("**Crop Paths:**")
            for i, crop in enumerate(test_case['retrieved_crops'], 1):
                st.code(f"{i}. {crop['crop_path']}", language=None)
    
    st.markdown("---")
    
    # Fila 2: Retrieved Crops
    display_retrieved_crops(test_case['retrieved_crops'])
    
    st.markdown("---")
    
    # Fila 3: Generated Answer
    answer_display = test_case['generated_answer']
    if not show_full_answer and len(answer_display) > 500:
        answer_display = answer_display[:500] + "...\n\n*(Truncated. Enable 'Show Full Answer' to see complete response)*"
    
    display_generated_answer(
        answer_display,
        test_case['recall_at_k'],
        {
            'retrieval_time_ms': test_case['retrieval_time_ms'],
            'generation_time_ms': test_case['generation_time_ms']
        }
    )
    
    # Footer con atajos de teclado
    st.markdown("---")
    st.caption("💡 **Tip:** Use the ◀ Prev / Next ▶ buttons in the sidebar to navigate, or type a case number directly!")
    st.caption("🔧 RAG Multimodal Visualizer | Built with Streamlit")

if __name__ == "__main__":
    main()

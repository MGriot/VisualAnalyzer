import streamlit as st
import argparse
import os
import sys
from pathlib import Path
import tempfile
import cv2

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.pipeline import run_analysis
from src.color_analysis.project_manager import ProjectManager
from src import config

# --- Page Configuration ---
st.set_page_config(
    page_title="Visual Analyzer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---
@st.cache_data
def get_project_list():
    try:
        return ProjectManager().list_projects()
    except Exception as e:
        st.sidebar.error(f"Error loading projects: {e}")
        return []

# --- Session State Initialization ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- Sidebar / Controls ---
st.sidebar.title("🧮 Visual Analyzer")
st.sidebar.header("Configuration")

# Get debug mode from URL query params only
debug_mode = "debug" in st.query_params and st.query_params.debug[0] == 'true'

project_list = get_project_list()
if not project_list:
    st.sidebar.warning("No projects found. Please create a project first.")
    st.stop()

selected_project = st.sidebar.selectbox(
    "Select Project",
    options=project_list,
    index=0,
    help="Choose the project to run the analysis on."
)

uploaded_image = st.sidebar.file_uploader(
    "Upload Sample Image", 
    type=["png", "jpg", "jpeg"],
    help="Select the main image for the analysis."
)

# --- UI Logic for Debug and Non-Debug Modes ---
if debug_mode:
    st.sidebar.info("Debug mode is ON.")
    with st.sidebar.expander("Analysis Steps", expanded=True):
        run_color_alignment = st.checkbox("Color Alignment", value=True)
        run_geometrical_alignment = st.checkbox("Geometrical Alignment", value=True)
        run_object_alignment = st.checkbox("Object Alignment", value=True)
        run_apply_mask = st.checkbox("Apply Mask", value=True)
        run_blur = st.checkbox("Blur Image", value=True)
        run_aggregate = st.checkbox("Aggregate Matched Pixels", value=True)
        run_symmetry = st.checkbox("Symmetry Analysis", value=True)
    
    if run_color_alignment:
        uploaded_color_checker = st.sidebar.file_uploader(
            "Upload Color Checker Image", 
            type=["png", "jpg", "jpeg"],
            help="Required if Color Alignment is enabled."
        )
    else:
        uploaded_color_checker = None

    with st.sidebar.expander("Advanced Options", expanded=True):
        st.write("**Masking Options**")
        mask_bg_is_white = st.checkbox("Treat White as Mask Background", value=False)
        masking_order = st.text_input("Masking Order", value="1-2-3")
        
        st.write("**Blur Options**")
        blur_kernel_str = st.text_input("Blur Kernel (W H)", placeholder="e.g., 5 5")

        st.write("**Aggregation Options**")
        agg_kernel_size = st.number_input("Agg. Kernel Size", min_value=1, step=2, value=7)
        agg_min_area = st.number_input("Agg. Min Area Ratio", min_value=0.0, max_value=1.0, value=0.0005, step=0.0001, format="%.4f")
        agg_density_thresh = st.number_input("Agg. Density Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        st.write("**Reporting**")
        report_type = st.selectbox("Report Type", ["all", "html", "reportlab"], index=0)
else:
    # NORMAL MODE UI
    run_color_alignment = True
    uploaded_color_checker = st.sidebar.file_uploader(
        "Upload Color Checker Image", 
        type=["png", "jpg", "jpeg"],
        help="Required for analysis."
    )
    run_geometrical_alignment, run_object_alignment, run_apply_mask, run_blur, run_aggregate, run_symmetry = (True,) * 6
    mask_bg_is_white, masking_order, blur_kernel_str, agg_kernel_size, agg_min_area, agg_density_thresh, report_type = (False, "1-2-3", "", 7, 0.0005, 0.5, "reportlab")

# --- Run Button ---
if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
    if uploaded_image is None:
        st.sidebar.error("Please upload a sample image.")
    elif run_color_alignment and uploaded_color_checker is None:
        st.sidebar.error("Please upload a color checker image.")
    else:
        with st.spinner("Running analysis... This may take a moment."):
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = os.path.join(tmpdir, uploaded_image.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())

                cc_path = None
                if uploaded_color_checker:
                    cc_path = os.path.join(tmpdir, uploaded_color_checker.name)
                    with open(cc_path, "wb") as f:
                        f.write(uploaded_color_checker.getbuffer())

                args = argparse.Namespace(
                    project=selected_project, image=image_path, video=None, camera=False, drawing=None,
                    debug=debug_mode, color_alignment=run_color_alignment, sample_color_checker=cc_path,
                    alignment=run_geometrical_alignment, object_alignment=run_object_alignment,
                    apply_mask=run_apply_mask, blur=run_blur, aggregate=run_aggregate, symmetry=run_symmetry,
                    mask_bg_is_white=mask_bg_is_white, masking_order=masking_order, agg_kernel_size=agg_kernel_size,
                    agg_min_area=agg_min_area, agg_density_thresh=agg_density_thresh, report_type=report_type
                )
                
                try:
                    if blur_kernel_str:
                        w, h = map(int, blur_kernel_str.split())
                        if w % 2 == 0 or h % 2 == 0:
                            st.error("Blur kernel dimensions must be odd.")
                            st.stop()
                        args.blur_kernel = [w, h]
                    else:
                        args.blur_kernel = None
                except ValueError:
                    st.error("Invalid blur kernel format. Use 'W H' with a space.")
                    st.stop()

                results = run_analysis(args)
                st.session_state.analysis_results = results
        
        if st.session_state.analysis_results:
            st.success("Analysis complete!")
        else:
            st.error("Analysis failed. Check the console for error messages.")

# --- Main Page / Results Display ---
st.title("Analysis Results")

if st.session_state.analysis_results is None:
    st.info("Upload an image and click \"Run Analysis\" to see the results here.")
else:
    results = st.session_state.analysis_results
    raw_results = results.get("analysis_results_raw", {})
    
    # --- Download Button for PDF Report ---
    project_output_dir = Path(config.OUTPUT_DIR) / results.get("project_name", "")
    if project_output_dir.exists():
        list_of_files = list(project_output_dir.glob('*.pdf'))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            with open(latest_file, "rb") as f:
                st.sidebar.download_button(
                    label="Download Report (PDF)",
                    data=f,
                    file_name=latest_file.name,
                    mime="application/pdf",
                    use_container_width=True,
                )

    # --- Key Metrics ---
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Percentage Matched", f"{raw_results.get('percentage', 0):.2f}%")
    col2.metric("Matched Pixels", f"{raw_results.get('matched_pixels', 0):,}")
    col3.metric("Total Pixels", f"{raw_results.get('total_pixels', 0):,}")

    # --- Image Results ---
    st.subheader("Image Results")
    
    def get_full_path(relative_path_key):
        relative_path = results.get(relative_path_key)
        if not relative_path:
            return None
        full_path = project_output_dir / relative_path
        return str(full_path) if full_path.is_file() else None

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(get_full_path('image_path'), caption="Original Image", use_container_width=True)
    with img_col2:
        st.image(get_full_path('analyzed_image_path'), caption="Analyzed Image (Matched Pixels)", use_container_width=True)

    if debug_mode and results.get("debug_data", {}).get("image_pipeline"):
        st.subheader("Debug Image Pipeline")
        pipeline_images = results["debug_data"]["image_pipeline"]
        
        for i in range(1, len(pipeline_images), 2):
            cols = st.columns(2)
            step1 = pipeline_images[i]
            path1 = project_output_dir / step1['path']
            if path1.is_file():
                cols[0].image(str(path1), caption=step1['title'], use_container_width=True)
            
            if i + 1 < len(pipeline_images):
                step2 = pipeline_images[i+1]
                path2 = project_output_dir / step2['path']
                if path2.is_file():
                    cols[1].image(str(path2), caption=step2['title'], use_container_width=True)

import streamlit as st
import sys
import json
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import base64
import io

# --- Monkey-patch for Streamlit/Drawable-Canvas incompatibility ---
# The `image_to_url` function was removed in recent Streamlit versions,
# but streamlit-drawable-canvas still depends on it. We recreate it here.
from streamlit.elements import image as st_image

def patched_image_to_url(image, width=None, height=None, clamp=False, channels="RGB", output_format="auto", image_id=""):
    buffered = io.BytesIO()
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Apply the patch
st_image.image_to_url = patched_image_to_url
# --- End of Patch ---

# Add project root to path for module imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.color_analysis.project_manager import ProjectManager

# --- Page Configuration ---
st.set_page_config(page_title="Manage Dataset", page_icon="🎨")
st.title("🎨 Manage Project Dataset")
st.markdown("""
Select a project, then click on its training images to define the color space for analysis.
Your selections are saved automatically when you navigate between images.
""")

# --- Helper Functions ---
@st.cache_data
def get_project_list():
    return ProjectManager().list_projects()

def load_project_data(project_name):
    try:
        project_files = ProjectManager().get_project_file_paths(project_name)
        st.session_state.image_paths = [cfg['path'] for cfg in project_files.get("training_image_configs", [])]
        config_file_path = project_root / "data" / "projects" / project_name / "dataset_item_processing_config.json"
        st.session_state.config_file_path = config_file_path
        if config_file_path.exists():
            with open(config_file_path, 'r') as f:
                config_data = json.load(f)
            st.session_state.points_config = {cfg["filename"]: cfg.get("points", []) for cfg in config_data.get("image_configs", [])}
        else:
            st.session_state.points_config = {}
    except Exception as e:
        st.error(f"Error loading project data for '{project_name}'. Error: {e}")
        st.session_state.image_paths, st.session_state.points_config = [], {}

def reset_state_on_project_change():
    st.session_state.current_index = 0
    st.session_state.image_paths = []
    st.session_state.points_config = {}

# --- Session State Initialization ---
for key in ['current_index', 'image_paths', 'points_config']:
    if key not in st.session_state:
        reset_state_on_project_change()

# --- Main UI ---
project_list = get_project_list()
if not project_list:
    st.warning("No projects found. Please create a project first.")
else:
    selected_project = st.selectbox(
        "Select Project to Manage",
        options=project_list, index=0,
        on_change=reset_state_on_project_change
    )

    if selected_project:
        if not st.session_state.image_paths:
            load_project_data(selected_project)

        if not st.session_state.image_paths:
            st.warning("This project has no training images. Add images to the 'dataset/training' folder.")
        else:
            total_images = len(st.session_state.image_paths)
            current_path = st.session_state.image_paths[st.session_state.current_index]
            image_name = current_path.name

            st.subheader(f"Image {st.session_state.current_index + 1}/{total_images}: {image_name}")

            bg_image = Image.open(current_path)

            initial_points = st.session_state.points_config.get(image_name, [])
            initial_drawing = {"objects": []}
            for p in initial_points:
                initial_drawing["objects"].append({
                    "type": "circle", "left": p['x'], "top": p['y'],
                    "radius": p.get('radius', 7), "fill": "rgba(255, 0, 0, 0.7)", "stroke": "red"
                })

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.7)",
                stroke_width=2,
                stroke_color="red",
                background_image=bg_image, # Pass the PIL image directly
                update_streamlit=True,
                height=bg_image.height,
                width=bg_image.width,
                drawing_mode="point",
                initial_drawing=initial_drawing,
                key=f"canvas_{st.session_state.current_index}"
            )

            if canvas_result.json_data and canvas_result.json_data["objects"]:
                points = []
                for obj in canvas_result.json_data["objects"]:
                    if obj['type'] == 'circle':
                        points.append({"x": int(obj['left']), "y": int(obj['top']), "radius": int(obj['radius'])})
                st.session_state.points_config[image_name] = points
            else:
                 st.session_state.points_config[image_name] = []

            col1, col2, col3, col4 = st.columns([2,2,2,3])
            if col1.button("⬅️ Previous"):
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
                    st.rerun()
            
            if col2.button("Next ➡️"):
                if st.session_state.current_index < total_images - 1:
                    st.session_state.current_index += 1
                    st.rerun()

            if col3.button("🗑️ Clear Points"):
                st.session_state.points_config[image_name] = []
                st.rerun()

            if col4.button("💾 Save Configuration to File", type="primary"):
                image_configs = []
                for img_name, points in st.session_state.points_config.items():
                    if points:
                        image_configs.append({"filename": img_name, "method": "points", "points": points})
                
                config_data = {"image_configs": image_configs}
                with open(st.session_state.config_file_path, 'w') as f:
                    json.dump(config_data, f, indent=4)
                st.success(f"Configuration saved to {st.session_state.config_file_path.name}!")
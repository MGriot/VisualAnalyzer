import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.create_project import create_project

st.set_page_config(page_title="Create Project", page_icon="📂")

st.title("📂 Create New Project")

st.markdown("""
Use this tool to scaffold a new project directory. This will create the necessary folders (`dataset`, `samples`, etc.) and default configuration files.
""")

project_name = st.text_input(
    "Enter New Project Name",
    placeholder="e.g., new_product_line"
)

if st.button("Create Project", type="primary"):
    if not project_name:
        st.error("Please enter a project name.")
    else:
        with st.spinner(f"Creating project '{project_name}'..."):
            messages = create_project(project_name)
            
            output_str = "\n".join(messages)
            if "Error" in output_str:
                st.error(output_str)
            else:
                st.success(f"Project '{project_name}' created successfully!")
                st.code(output_str, language="bash")

"""
This is the main entry point for the Streamlit GUI of the Visual Analyzer application.

This script sets up the main page of the Streamlit application and provides an
overview of the available tools.
"""

import streamlit as st

st.set_page_config(
    page_title="Visual Analyzer - Home",
    page_icon="🏠",
    layout="wide"
)

st.title("Welcome to the Visual Analyzer! 👋")

st.markdown("""
This application is a suite of tools for advanced image analysis. 

**👈 Select a tool from the sidebar to get started.**

### Available Tools:

- **Run Analysis:** The main analysis pipeline. Upload an image and a color checker to perform a full analysis, including color correction, masking, and symmetry analysis.
- **Create Project:** Set up a new project directory with the required folder structure and default configuration files.
- **Manage Dataset:** Launch the desktop GUI to define the color space for a project by selecting sample points on training images.

""")

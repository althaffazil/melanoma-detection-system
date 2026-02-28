import sys
import os

# Add project root to Python path (for Streamlit Cloud)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from src.inference.predictor import SkinLesionPredictor

SYSTEM_NAME = "Automated Dermoscopic Melanoma Detection System"
BEST_THRESHOLD = 0.2527

CLASSIFIER_PATH = "checkpoints/classifier_best.pth"
SEGMENTATION_PATH = "checkpoints/segmentation_best.pth"

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title=SYSTEM_NAME,
    layout="wide"
)

st.title(SYSTEM_NAME)
st.markdown(
    "Deep learning–based melanoma detection using EfficientNet-B3 "
    "for classification and U-Net for lesion segmentation."
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("System Overview")
st.sidebar.write("**Classification Backbone:** EfficientNet-B3")
st.sidebar.write("**Segmentation Architecture:** U-Net")
st.sidebar.write(f"**Decision Threshold:** {BEST_THRESHOLD}")

st.sidebar.markdown("---")
st.sidebar.header("Dataset Citation")
st.sidebar.write(
    "Training dataset derived from the ISIC 2018 Challenge dataset "
    "(International Skin Imaging Collaboration)."
)

show_segmentation = st.sidebar.checkbox(
    "Display Segmentation Overlay",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.caption("Research prototype – not approved for clinical use.")

# -------------------------------------------------
# Load Predictor
# -------------------------------------------------
@st.cache_resource
def load_predictor():
    return SkinLesionPredictor(
        classifier_path=CLASSIFIER_PATH,
        segmentation_path=SEGMENTATION_PATH
    )

predictor = load_predictor()

# -------------------------------------------------
# PDF Report Generator
# -------------------------------------------------
def generate_pdf(prob, decision):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph(SYSTEM_NAME, styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    report_data = [
        ["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Classification Model", "EfficientNet-B3"],
        ["Segmentation Model", "U-Net"],
        ["Training Dataset", "ISIC 2018 Challenge Dataset"],
        ["Decision Threshold", f"{BEST_THRESHOLD}"],
        ["Melanoma Probability", f"{prob:.4f}"],
        ["Model Decision", decision],
    ]

    table = Table(report_data, colWidths=[2.5 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph(
        "Dataset Citation: ISIC 2018 Challenge Dataset – "
        "International Skin Imaging Collaboration.",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(
        "Medical Disclaimer: This system is intended for research "
        "and educational purposes only and does not provide medical diagnosis.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# -------------------------------------------------
# Main UI
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Running inference..."):

        prob = predictor.predict_classification(image_np)

        if show_segmentation:
            mask = predictor.predict_segmentation(image_np)

            mask_resized = cv2.resize(
                mask,
                (image_np.shape[1], image_np.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            overlay = image_np.copy()
            overlay[mask_resized > 0.5] = [255, 0, 0]
            blended = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)

    if show_segmentation:
        with col2:
            st.subheader("Segmentation Overlay")
            st.image(blended, use_container_width=True)

    st.markdown("---")
    st.subheader("Classification Output")

    st.metric("Melanoma Probability", f"{prob:.4f}")
    st.progress(int(prob * 100))

    if prob >= BEST_THRESHOLD:
        decision = "Positive (Melanoma Suggested)"
        st.error(decision)
    else:
        decision = "Negative (Non-Melanoma Suggested)"
        st.success(decision)

    pdf_buffer = generate_pdf(prob, decision)

    st.download_button(
        label="Download Diagnostic Report (PDF)",
        data=pdf_buffer,
        file_name="melanoma_detection_report.pdf",
        mime="application/pdf"
    )

    st.markdown("---")
    st.markdown(
        "### Dataset Reference\n"
        "This system was trained using the ISIC 2018 Challenge dataset "
        "from the International Skin Imaging Collaboration (ISIC)."
    )

    st.markdown(
        "**Disclaimer:** This application is a research prototype "
        "and is not an approved medical device."
    )
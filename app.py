# app.py — NeuroAI (Final Corrected Version with Robust Filtering)
"""
Streamlit app for tumor segmentation + volumetric analysis.
- Wide layout and st.columns for side-by-side comparison.
- Added metadata input for professional reports.
- Added chart for multi-slice area comparison.
- Robust filtering implemented: Analysis is only reported if the segmented area is greater 
  than MIN_AREA_CM2 AND the mean model confidence is above MIN_CONFIDENCE_FLOOR (0.50).
"""

import streamlit as st
from pathlib import Path
from datetime import datetime, date 
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background') 

# --- Configuration ---
MODEL_FILENAME = "meramodel(5).h5" 
SIZE = 128
MIN_CONFIDENCE_FLOOR = 0.80  # Only consider results with a mean confidence > 50%
MIN_AREA_CM2 = 0.05          # Disregard areas smaller than 0.05 cm²

# -----------------------------
# Utility Functions
# -----------------------------

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    return img.convert("RGB")

@st.cache_resource(show_spinner="Loading AI Model...")
def load_brain_ai(filename: str):
    model_path = Path(filename)
    if not model_path.exists():
        st.error(f"Model not found: {filename}")
        st.stop()
    try:
        model = load_model(str(model_path), compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


def preprocess(img: Image.Image, size: int = SIZE):
    img = ensure_rgb(img)
    w, h = img.size
    resized = img.resize((size, size))
    arr = np.array(resized).astype(np.float32) / 255.0
    return arr, (w, h), Image.fromarray((arr * 255).astype(np.uint8))


def postprocess(pred, orig_size, threshold=0.5):
    w, h = orig_size
    if pred.ndim == 3:
        pred = np.squeeze(pred)
    prob = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob > threshold).astype(np.uint8)
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask, prob


def calculate_area_volume(mask, scale_mm_per_px, thickness_mm):
    pixels = int(mask.sum())
    area_mm2 = pixels * (scale_mm_per_px ** 2)
    area_cm2 = area_mm2 / 100.0
    volume_cm3 = area_cm2 * (thickness_mm / 10.0)
    return pixels, area_cm2, volume_cm3


def make_overlay(img, mask, prob):
    overlay = np.array(ensure_rgb(img)).astype(np.uint8)
    neon_mask = np.zeros_like(overlay)
    neon_mask[mask == 1] = [0, 255, 136] 
    overlay = cv2.addWeighted(overlay, 0.6, neon_mask, 0.4, 0)
    prob_colored = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    prob_colored = cv2.cvtColor(prob_colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay), Image.fromarray(prob_colored)


def mean_confidence(pred, mask):
    mask_bool = (mask > 0)
    if mask_bool.sum() == 0:
        return 0.0
    return float(pred[mask_bool].mean())

def create_confidence_histogram(prob, mask):
    mask_bool = (mask > 0)
    if mask_bool.sum() == 0:
        return None
        
    confidence_values = prob[mask_bool]
    
    fig, ax = plt.subplots()
    ax.hist(confidence_values, bins=20, range=(0, 1), edgecolor='black', color='#00FF88')
    ax.set_title("Model Confidence Distribution (Tumor Pixels)")
    ax.set_xlabel("Probability Score")
    ax.set_ylabel("Pixel Count")
    ax.grid(axis='y', alpha=0.5)
    
    mean_conf = confidence_values.mean()
    ax.axvline(mean_conf, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_conf:.2f}')
    ax.legend()
    
    return fig


def detect_tumor(model, img, scale, thickness, threshold=0.5):
    arr, orig_size, resized_img = preprocess(img)
    
    # Run prediction
    pred = model.predict(arr[None, ...], verbose=0)
    pred_map = pred[0, :, :, 0] if pred.ndim == 4 else pred[0]
    
    # Post-process to get mask (NumPy array at original size) and raw probability map
    mask_np, prob = postprocess(pred_map, orig_size, threshold) 
    px, area, vol = calculate_area_volume(mask_np, scale, thickness)
    conf = mean_confidence(prob, mask_np)
    overlay, prob_img = make_overlay(img, mask_np, prob)

    return {
        "mask": Image.fromarray(mask_np * 255),
        "mask_np": mask_np,
        "overlay": overlay,
        "prob_map": prob_img,
        "prob_raw": prob,
        "cm2": area,
        "cm3_slice": vol,
        "px": px,
        "conf": conf,
        "original_img": img,
    }

# -----------------------------
# Streamlit App
# -----------------------------

def main():
    # Set wide layout and title/icon
    st.set_page_config(page_title="NeuroAI • Tumor Volume Calculator", page_icon="🧠", layout="wide")

    # --- Initialize Session State ---
    if 'slice_areas' not in st.session_state:
        st.session_state.slice_areas = []
        st.session_state.case_id = ""
        st.session_state.scan_date = date.today() 
    if 'total_volume_cm3' not in st.session_state:
        st.session_state.total_volume_cm3 = 0.0

    # --- Sidebar Controls ---
    with st.sidebar:
        st.title("NeuroAI Panel")
        st.markdown("---")
        
        st.subheader("Analysis Parameters")
        scale = st.slider("Pixel Scale (mm/px)", 0.1, 2.0, 0.5, 0.01)
        thickness = st.slider("Slice Thickness (mm)", 0.5, 10.0, 5.0, 0.1)
        threshold = st.slider("Segmentation Threshold", 0.1, 0.9, 0.5, 0.01)
        st.markdown(f"**Min Confidence Floor:** {MIN_CONFIDENCE_FLOOR:.1%}") 
        st.markdown(f"**Min Area Threshold:** {MIN_AREA_CM2:.2f} cm²")
        st.markdown("---")
        
        if st.button("Reset All Analysis", use_container_width=True):
            st.session_state.slice_areas = []
            st.session_state.total_volume_cm3 = 0.0
            st.session_state.case_id = ""
            st.session_state.scan_date = date.today() 
            st.experimental_rerun()
            
    # --- Main App Body ---
    st.title("NeuroAI: Tumor Volume Calculator for MRI Slices")

    # Patient/Case Metadata Expander
    with st.expander("📝 Enter Patient & Scan Metadata", expanded=True):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.session_state.case_id = st.text_input(
                "Patient ID / Case Ref", 
                value=st.session_state.case_id, 
                placeholder="e.g., PT-2025-001"
            )
        with col_m2:
            st.session_state.scan_date = st.date_input(
                "Scan Date", 
                value=st.session_state.scan_date
            )
            
    # Display Total Volume Status
    if st.session_state.total_volume_cm3 > 0:
        st.metric(
            "Total Estimated Tumor Volume", 
            f"{st.session_state.total_volume_cm3:.3f} cm³",
            f"Across {len(st.session_state.slice_areas)} slices"
        )
    
    st.markdown("---")
    
    # --- Model Loading and Upload ---
    model = load_brain_ai(MODEL_FILENAME)
    
    uploaded = st.file_uploader("Upload MRI Slice (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        
        with st.spinner("🧠 Analyzing MRI slice..."):
            result = detect_tumor(model, img, scale, thickness, threshold)
        
        # --- CONDITIONAL CHECK WITH ROBUST FILTERS ---
        is_tumor_detected = result['cm2'] > MIN_AREA_CM2  # Check against min area (0.05 cm²)
        is_confident_enough = result['conf'] >= MIN_CONFIDENCE_FLOOR # Check against min confidence (50%)

        if is_tumor_detected and is_confident_enough:
            st.success("Analysis Complete! Tumor Detected.")
            
            # Update running volume total
            st.session_state.slice_areas.append(result['cm2'])
            st.session_state.total_volume_cm3 = sum(st.session_state.slice_areas) * (thickness / 10.0)

            
            # --- Side-by-Side Visualization ---
            st.subheader("Visual Comparison")
            col_vis1, col_vis2 = st.columns(2)
            
            with col_vis1:
                st.caption("Original Scan")
                st.image(result['original_img'], use_column_width=True)
                
            with col_vis2:
                st.caption("Segmentation Overlay (Tumor highlighted in Neon Green)")
                st.image(result['overlay'], use_column_width=True)

            st.markdown("---")
            
            # --- Advanced Visualization Tabs ---
            tabs = st.tabs(["📊 Slice Area History", "🔬 Confidence Histogram", "☁️ Probability Map", "📋 Report Details"])

            # Tab 0: Slice Area History
            with tabs[0]:
                if st.session_state.slice_areas:
                    st.subheader(f"Area History ({len(st.session_state.slice_areas)} Slices)")
                    df_slices = pd.DataFrame({
                        'Slice': [f"Slice {i+1}" for i in range(len(st.session_state.slice_areas))],
                        'Area (cm²)': st.session_state.slice_areas
                    })
                    st.bar_chart(df_slices.set_index('Slice'))
                    st.caption("Visualization of the tumor area calculated for each uploaded slice.")
                else:
                    st.info("Upload more slices to start comparing area history!")
                    
            # Tab 1: Confidence Histogram 
            with tabs[1]:
                st.subheader("Model Certainty on Tumor Pixels")
                prob_mask = result['mask_np']
                
                hist_fig = create_confidence_histogram(result['prob_raw'], prob_mask)
                
                if hist_fig:
                    st.pyplot(hist_fig)
                    st.caption("Distribution of the model's prediction probability for pixels included in the final tumor mask.")
                    plt.close(hist_fig) 
                else:
                    st.warning("No tumor detected in this slice to generate a confidence histogram.")

            # Tab 2: Probability Map
            with tabs[2]:
                st.subheader("Raw Probability Map")
                st.image(result['prob_map'], caption="Pixel-wise Probability Map (Heatmap style)", use_column_width=True)

            # Tab 3: Report Details
            with tabs[3]:
                st.header("Volumetric Report for Current Slice")
                col_rep1, col_rep2 = st.columns(2)
                
                with col_rep1:
                    st.markdown(f"**Patient/Case ID:** `{st.session_state.case_id or 'N/A'}`")
                    st.markdown(f"**Scan Date:** `{st.session_state.scan_date.strftime('%Y-%m-%d')}`") 
                    st.markdown(f"**Analysis Time:** `{datetime.now():%Y-%m-%d %H:%M:%S}`")
                    st.markdown(f"---")
                    st.metric("Current Slice Area", f"{result['cm2']:.3f} cm²")
                    st.metric("Slice Volume (Estimate)", f"{result['cm3_slice']:.3f} cm³")
                    st.metric("Mean Model Confidence", f"{result['conf']:.1%}")

                with col_rep2:
                    st.markdown(f"**Segmentation Threshold:** `{threshold}`")
                    st.markdown(f"**Pixel Scale:** `{scale} mm/px`")
                    st.markdown(f"**Slice Thickness:** `{thickness} mm`")
                    st.markdown(f"---")
                    st.metric("Total Tumor Volume", f"{st.session_state.total_volume_cm3:.3f} cm³", delta="Cumulative across all uploads.")
                    st.markdown(f"**Total Pixels Segmented:** `{result['px']:,}`")

                # Download Button
                report_content = f"""
                NeuroAI Volumetric Report - {datetime.now():%Y-%m-%d %H:%M:%S}
                ------------------------------------------------------------
                Patient ID: {st.session_state.case_id or 'N/A'}
                Scan Date: {st.session_state.scan_date.strftime('%Y-%m-%d')}
                
                --- Volumetric Summary ---
                Total Cumulative Volume: {st.session_state.total_volume_cm3:.3f} cm³ ({len(st.session_state.slice_areas)} slices)
                
                --- Current Slice Data ---
                Slice Area: {result['cm2']:.3f} cm²
                Slice Volume: {result['cm3_slice']:.3f} cm³
                Mean Confidence: {result['conf']:.1%}
                
                --- Parameters ---
                Segmentation Threshold: {threshold}
                Pixel Scale: {scale} mm/px
                Slice Thickness: {thickness} mm
                ------------------------------------------------------------
                Disclaimer: Research use only. Not for clinical use.
                """
                st.download_button("Download Full Report (.txt)", report_content, "neuroai_volumetric_report.txt", use_container_width=True)

        else:
            # Handle cases where area is too small OR confidence is too low
            st.error("No significant Tumor Found in this slice OR Model Confidence is too low to report.")
            st.info(
                f"The analysis found {result['cm2']:.3f} cm² of area (Required: > {MIN_AREA_CM2:.2f} cm²). "
                f"The mean model confidence was **{result['conf']:.1%}** (Required: > {MIN_CONFIDENCE_FLOOR:.1%})."
            )
            st.image(result['original_img'], caption="Original Scan - Analysis Failed to Meet Robust Criteria", use_column_width=False)


    st.markdown("---")
    st.caption("Developed by Affan • Utilizing Streamlit, TensorFlow, and OpenCV")


if __name__ == '__main__':
    main()
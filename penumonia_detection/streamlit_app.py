import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pydicom as dcm
import cv2
from PIL import Image
import io
from pathlib import Path

# Define the path to the saved best model (relative to the app's root)
MODEL_PATH = Path(__file__).resolve().parent.parent / 'best_fine_tuned_resnet50_model.keras'

# Define the focal loss function (must be available when loading the model)
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(tf.keras.backend.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.keras.backend.equal(y_true, 1), alpha, 1 - alpha)
        cross_entropy = -tf.keras.backend.log(p_t)
        loss = alpha_factor * tf.keras.backend.pow((1 - p_t), gamma) * cross_entropy
        return tf.keras.backend.mean(loss, axis=-1)
    return focal_loss_fixed

@st.cache_resource
def load_pneumonia_model():
    """Loads the fine-tuned ResNet50 model."""
    try:
        model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25)})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def preprocess_image(image_bytes, target_size=(224, 224), n_channels=3):
    """Preprocesses the uploaded image for model inference."""
    try:
        # Attempt to read as DICOM first
        try:
            dicom_data = dcm.dcmread(io.BytesIO(image_bytes), force=True)
            if "PixelData" not in dicom_data:
                raise dcm.errors.InvalidDicomError("DICOM file has no pixel data")
            img_array = dicom_data.pixel_array

            if img_array.ndim == 4:
                img_array = img_array[0, ..., 0]
            elif img_array.ndim == 3:
                if img_array.shape[-1] <= 4: # Assume (H,W,C)
                    img_array = img_array[:, :, 0]
                elif img_array.shape[0] <= 4: # Assume (C,H,W)
                    img_array = img_array[0, :, :]
                else: # Assume (D,H,W)
                    img_array = img_array[0, :, :]
            img_array = np.squeeze(img_array)

            if img_array.ndim != 2:
                raise ValueError("DICOM image array is not 2D after processing.")

            # Normalize DICOM pixel values to 0-255 range
            if img_array.dtype != np.float32:
                img_array = img_array.astype(np.float32)
            if img_array.max() > 0:
                img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0
            else:
                img_array = np.zeros_like(img_array)
            img_array = img_array.astype(np.uint8) # Convert to uint8 for OpenCV

        except (dcm.errors.InvalidDicomError, EOFError) as dicom_error:
            # If not DICOM, try as a regular image (PNG, JPEG, etc.)
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert to grayscale
            except (OSError, ValueError) as image_error:
                raise ValueError(
                    "Unable to decode this file as DICOM "
                    f"({type(dicom_error).__name__}: {dicom_error}) or as PNG/JPEG "
                    f"({type(image_error).__name__}: {image_error})."
                ) from image_error
            img_array = np.array(img)

        # Resize using OpenCV for better quality
        resized_img = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)

        # Add channel dimension(s)
        if n_channels == 1:
            final_img_array = np.expand_dims(resized_img, axis=-1) # (H, W, 1)
        elif n_channels == 3:
            final_img_array = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB) # (H, W, 3)
        else:
            raise ValueError(f"Unsupported number of channels: {n_channels}")

        # Normalize to 0-1 range (float32)
        final_img_array = final_img_array.astype(np.float32) / 255.0

        return final_img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Streamlit app layout
st.set_page_config(page_title="PulmoVision", layout="wide")

st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="margin-bottom: 0.2em;">PulmoVision</h1>
        <h4 style="margin-top: 0; color: #666;">Pneumonia Detection from Chest X-ray Images</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Upload a chest X-ray image (DICOM, PNG, or JPG) to predict if it shows signs of Lung Opacity.")

# Load the model
model = load_pneumonia_model()

if model is not None:
    st.markdown(
        """
        <div style="margin: 1rem 0 1.5rem 0; padding: 0.9rem 1rem; background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%); border: 1px solid #dbeafe; border-radius: 12px; max-width: 520px;">
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;">
                <div>
                    <div style="font-size: 0.76rem; letter-spacing: 0.08em; text-transform: uppercase; color: #475569; font-weight: 700;">Model Status</div>
                    <div style="font-size: 1rem; font-weight: 700; color: #0f172a; margin-top: 4px;">PulmoVision Ready</div>
                </div>
                <span style="display: inline-flex; align-items: center; padding: 6px 10px; background: #ecfdf5; color: #166534; border: 1px solid #a7f3d0; border-radius: 999px; font-size: 0.8rem; font-weight: 700;">● Active</span>
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.9rem; color: #334155;">
                Threshold: <strong>0.30</strong> · Input types: <strong>DICOM, PNG, JPG</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["dcm", "png", "jpg", "jpeg"]) # Allow DICOM uploads

    if uploaded_file is not None:
        # Read file as bytes
        image_bytes = uploaded_file.getvalue()

        processed_image = preprocess_image(image_bytes, target_size=(224, 224), n_channels=3)

        if processed_image is not None:
            st.image(processed_image, caption='Uploaded Image', width=300)
            st.write("")
            st.write("Classifying...")

            # Add batch dimension for prediction
            input_tensor = np.expand_dims(processed_image, axis=0)

            # Make prediction
            prediction_prob = model.predict(input_tensor, verbose=0)[0][0]

            # Define the optimized threshold
            threshold = 0.3
            opacity_percentage = prediction_prob * 100
            confidence = max(prediction_prob, 1 - prediction_prob) * 100

            # Classify based on threshold
            if prediction_prob >= threshold:
                badge_bg = "#fdecea"
                badge_color = "#b42318"
                label = "Lung Opacity (Pneumonia Detected)"
                icon = "⚠️"
            else:
                badge_bg = "#ecfdf3"
                badge_color = "#027a48"
                label = "Normal (No Lung Opacity Detected)"
                icon = "✅"

            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 12px; margin-top: 1rem; flex-wrap: wrap;">
                    <span style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; background-color: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}; border-radius: 10px; font-weight: 700; font-size: 0.95rem; box-shadow: 0 1px 2px rgba(16,24,40,0.06);">
                        {icon} {label}
                    </span>
                    <span style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; background-color: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; border-radius: 10px; font-weight: 600; font-size: 0.92rem; box-shadow: 0 1px 2px rgba(16,24,40,0.04);">
                        Lung Opacity Probability: {opacity_percentage:.1f}%
                    </span>
                    <span style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; background-color: #f8fafc; color: #334155; border: 1px solid #cbd5e1; border-radius: 10px; font-weight: 600; font-size: 0.92rem; box-shadow: 0 1px 2px rgba(16,24,40,0.04);">
                        Confidence: {confidence:.1f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(f"*Note: A lung opacity probability above {threshold * 100:.0f}% is classified as Lung Opacity. Confidence reflects how far the predicted probability is from 50%.*")
else:
    st.warning("Model could not be loaded. Please check the model path and file.")

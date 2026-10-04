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
    <style>
        .main > div {
            background: linear-gradient(135deg, rgba(239,246,255,0.88), rgba(248,250,252,0.96));
        }
        .stApp {
            background:
                linear-gradient(rgba(255,255,255,0.45), rgba(255,255,255,0.45)),
                radial-gradient(circle at 20% 20%, rgba(96,165,250,0.18), transparent 22%),
                radial-gradient(circle at 80% 30%, rgba(148,163,184,0.10), transparent 18%),
                linear-gradient(135deg, #f8fafc 0%, #eff6ff 55%, #f8fafc 100%);
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(148, 163, 184, 0.18) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148, 163, 184, 0.18) 1px, transparent 1px),
                radial-gradient(circle at center, rgba(15, 23, 42, 0.08) 0%, rgba(15, 23, 42, 0.02) 35%, transparent 70%),
                radial-gradient(ellipse at center, rgba(96, 165, 250, 0.12) 0%, rgba(96, 165, 250, 0) 55%);
            background-size: 28px 28px, 28px 28px, 100% 100%, 100% 100%;
            opacity: 0.42;
            pointer-events: none;
            z-index: 0;
        }
        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            background:
                url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='700' height='420' viewBox='0 0 700 420'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cpath d='M240 90c-18 0-32 14-32 32v174c0 18 14 32 32 32h220c18 0 32-14 32-32V122c0-18-14-32-32-32H240zm52 38h116c20 0 36 16 36 36v108c0 20-16 36-36 36H292c-20 0-36-16-36-36V164c0-20 16-36 36-36z' fill='rgba(59,130,246,0.09)'/%3E%3Cpath d='M292 164h116v108H292z' fill='rgba(59,130,246,0.06)'/%3E%3Cpath d='M210 142c16 0 28 12 28 28v82c0 16-12 28-28 28s-28-12-28-28v-82c0-16 12-28 28-28zm280 0c16 0 28 12 28 28v82c0 16-12 28-28 28s-28-12-28-28v-82c0-16 12-28 28-28zM330 90v240M370 90v240M290 206h120M290 246h120M290 286h120' stroke='rgba(30,41,59,0.10)' stroke-width='6' stroke-linecap='round'/%3E%3C/g%3E%3C/svg%3E") center/cover no-repeat;
            opacity: 0.7;
            pointer-events: none;
            z-index: 0;
        }
        .stApp > * {
            position: relative;
            z-index: 1;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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
            # Add batch dimension for prediction
            input_tensor = np.expand_dims(processed_image, axis=0)

            # Make prediction
            prediction_prob = model.predict(input_tensor, verbose=0)[0][0]

            # Define the optimized threshold
            threshold = 0.3
            opacity_percentage = prediction_prob * 100
            confidence = max(prediction_prob, 1 - prediction_prob) * 100

            if opacity_percentage >= 70:
                risk_level = "High"
                risk_bg = "#fee2e2"
                risk_color = "#b91c1c"
            elif opacity_percentage >= 40:
                risk_level = "Moderate"
                risk_bg = "#fef3c7"
                risk_color = "#b45309"
            else:
                risk_level = "Low"
                risk_bg = "#dcfce7"
                risk_color = "#15803d"

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

            col1, col2 = st.columns([1.1, 1.5])

            with col1:
                st.markdown(
                    """
                    <div style="background: linear-gradient(135deg, #f8fafc 0%, #edf2ff 100%); border: 1px solid #dbeafe; border-radius: 16px; padding: 1rem; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);">
                        <div style="font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: #475569; font-weight: 700; margin-bottom: 0.5rem;">Uploaded Image</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(processed_image, width=300)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.2rem; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);">
                        <div style="font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: #475569; font-weight: 700; margin-bottom: 0.8rem;">Prediction Summary</div>
                        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1rem;">
                            <span style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; background-color: {badge_bg}; color: {badge_color}; border: 1px solid {badge_color}; border-radius: 10px; font-weight: 700; font-size: 0.95rem;">
                                {icon} {label}
                            </span>
                        </div>
                        <div style="display: grid; gap: 0.8rem;">
                            <div style="padding: 0.8rem 0.9rem; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px; color: #1d4ed8; font-weight: 600;">
                                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                                    <span>Lung Opacity Probability</span>
                                    <strong>{opacity_percentage:.1f}%</strong>
                                </div>
                                <div style="height: 10px; width: 100%; background: #dbeafe; border-radius: 999px; overflow: hidden;">
                                    <div style="height: 100%; width: {opacity_percentage:.1f}%; background: linear-gradient(90deg, #60a5fa 0%, #2563eb 100%); border-radius: inherit;"></div>
                                </div>
                            </div>
                            <div style="padding: 0.8rem 0.9rem; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; color: #334155; font-weight: 600;">
                                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                                    <span>Confidence</span>
                                    <strong>{confidence:.1f}%</strong>
                                </div>
                                <div style="height: 10px; width: 100%; background: #e2e8f0; border-radius: 999px; overflow: hidden;">
                                    <div style="height: 100%; width: {confidence:.1f}%; background: linear-gradient(90deg, #94a3b8 0%, #475569 100%); border-radius: inherit;"></div>
                                </div>
                            </div>
                            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.7rem 0.8rem; background: {risk_bg}; border: 1px solid {risk_color}; border-radius: 10px; color: {risk_color}; font-weight: 700;">
                                <span>Risk Level</span>
                                <span>{risk_level}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write(f"*Note: A lung opacity probability above {threshold * 100:.0f}% is classified as Lung Opacity. Confidence reflects how far the predicted probability is from 50%.*")
else:
    st.warning("Model could not be loaded. Please check the model path and file.")

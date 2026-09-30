# Pneumonia-Detection
Capstone project on Pneumonia Detection

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub, including the Git LFS model files. Confirm that Git LFS is enabled for the repository so the model is available during deployment.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from this repository and select the `main` branch.
3. Set the main file path to `penumonia_detection/streamlit_app.py` and deploy. Streamlit installs dependencies from `penumonia_detection/requirements.txt`.

The app loads `best_fine_tuned_resnet50_model.keras` from the repository root. Uploaded images are processed for inference and are not saved by the app. This model output is for demonstration only and is not a medical diagnosis.

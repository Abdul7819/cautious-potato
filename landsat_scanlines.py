import streamlit as st
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Function to load the image from an in-memory file object with optional downsampling
def load_image(file, downsample_factor=2):
    try:
        with MemoryFile(file) as memfile:
            with memfile.open() as src:
                # Downsample image if factor is greater than 1
                out_shape = (src.count, src.height // downsample_factor, src.width // downsample_factor)
                image = src.read(
                    out_shape=out_shape,
                    resampling=Resampling.bilinear
                )
                image = image.transpose(1, 2, 0)  # Convert to HWC format for OpenCV
                return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to identify and correct bad pixels
def correct_bad_pixels(image):
    corrected_image = np.copy(image)
    mask = image == 0  # Assuming 0 as the bad pixel value
    corrected_image[mask] = np.mean(image[~mask], axis=0)
    return corrected_image

# Function to apply classifier in chunks to avoid memory issues
def apply_classifier_in_chunks(image, classifier_type, chunk_size=10000):
    h, w, c = image.shape
    image_flat = image.reshape(-1, c)

    if classifier_type == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=100)
    elif classifier_type == 'SVM':
        classifier = SVC()
    elif classifier_type == 'KNN':
        classifier = KNeighborsClassifier()

    # Simple binary classification example based on intensity
    labels = np.where(np.mean(image_flat, axis=1) > 128, 1, 0)
    
    # Process in chunks
    classified_image_flat = np.zeros(image_flat.shape[0], dtype=np.uint8)
    
    for start in range(0, image_flat.shape[0], chunk_size):
        end = start + chunk_size
        classifier.fit(image_flat[start:end], labels[start:end])
        predictions = classifier.predict(image_flat[start:end])
        classified_image_flat[start:end] = predictions

    classified_image = classified_image_flat.reshape(h, w)
    
    return classified_image

# Streamlit app layout
st.title("Landsat 7 Image Correction and Classification")

uploaded_file = st.file_uploader("Upload a Landsat 7 Image", type=["tif", "tiff"])

if uploaded_file:
    downsample_factor = st.slider("Downsample Factor", 1, 10, 2)  # User can adjust downsample factor
    image = load_image(uploaded_file, downsample_factor=downsample_factor)
    
    if image is not None:
        st.image(image, caption="Original Image", use_column_width=True)

        classifier_type = st.selectbox("Choose Classifier", ['Random Forest', 'SVM', 'KNN'])
        
        if st.button("Apply Classifier and Correct Image"):
            classified_image = apply_classifier_in_chunks(image, classifier_type)
            corrected_image = correct_bad_pixels(classified_image)
            
            st.image(corrected_image, caption="Corrected Image", use_column_width=True)
            # Provide download button for corrected image
            corrected_image_bytes = corrected_image.tobytes()
            st.download_button("Download Corrected Image", data=corrected_image_bytes, file_name="corrected_image.tif")
    else:
        st.error("Failed to load the image. Please ensure the file is a valid GeoTIFF.")

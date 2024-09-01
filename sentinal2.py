import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from io import BytesIO

def calculate_ndvi(band_red, band_nir):
    """Calculate Normalized Difference Vegetation Index (NDVI)."""
    ndvi = (band_nir - band_red) / (band_nir + band_red + 1e-10)
    return ndvi

def classify_landcover(ndvi, threshold_water=0.1, threshold_buildup=0.3):
    """
    Classify landcover into water and buildup areas based on NDVI thresholds.
    - Water Landcover: NDVI < threshold_water
    - Buildup Landcover: NDVI > threshold_buildup
    """
    water_landcover = np.where(ndvi < threshold_water, 1, 0)
    buildup_landcover = np.where(ndvi > threshold_buildup, 1, 0)
    return water_landcover, buildup_landcover

def plot_index(index, title, colormap='RdYlGn', value_range=(-1.0, 1.0)):
    """Plot an index with a color map and display the range."""
    plt.figure(figsize=(10, 10))
    plt.imshow(index, cmap=colormap, vmin=value_range[0], vmax=value_range[1])
    plt.colorbar()
    plt.title(title)
    plt.xlabel(f"Range: {value_range[0]} to {value_range[1]}")
    st.pyplot(plt)

st.title("Sentinel-2 Image Analysis: NDVI, Water Bodies, and Buildup Landcover")

uploaded_file = st.file_uploader("Upload Sentinel-2 Image", type=["tif"])

if uploaded_file is not None:
    # Convert the uploaded file to a BytesIO object for rasterio to read
    with rasterio.open(BytesIO(uploaded_file.read())) as src:
        band_red = src.read(3)  # Assuming band 4 is red
        band_nir = src.read(4)  # Assuming band 8 is NIR

    # Calculate NDVI
    ndvi = calculate_ndvi(band_red, band_nir)

    # Classify landcover
    water_landcover, buildup_landcover = classify_landcover(ndvi)

    # Display NDVI with range
    st.write("**NDVI**")
    plot_index(ndvi, "NDVI", value_range=(-1.0, 1.0))

    # Display Water Landcover
    st.write("**Water Bodies (NDVI < 0.1)**")
    plot_index(water_landcover, "Water Landcover", colormap='Blues', value_range=(0, 1))

    # Display Buildup Landcover
    st.write("**Buildup Landcover (NDVI > 0.3)**")
    plot_index(buildup_landcover, "Buildup Landcover", colormap='Oranges', value_range=(0, 1))

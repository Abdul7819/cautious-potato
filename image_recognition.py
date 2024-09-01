import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f4;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#ff7f50,#ff6347);
        color: white;
    }
    .stButton > button {
        color: white;
        background-color: #ff7f50;
        border-radius: 10px;
        padding: 10px;
    }
    .stSpinner > div > div {
        color: #ff6347;
    }
    .stAlert > div {
        background-color: #ffefd5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for image upload and instructions
st.sidebar.title("Image Authenticity Checker")
st.sidebar.write("Upload an image to determine if it's real, AI-generated, or fake.")
uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to classify images
def classify_image(image):
    # Placeholder for actual model code
    model = models.resnet18(pretrained=True)
    model.eval()

    # Define transformations for the input image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply transformations and add a batch dimension
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(input_batch)

    # Placeholder logic for demo
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    _, prediction = torch.max(probabilities, 0)

    # Dummy classes for demonstration
    classes = ['Real', 'AI-generated', 'Fake']
    result = classes[prediction % len(classes)]

    return result

# Main page content
st.title("🌟 Image Authenticity Checker 🌟")
st.write("Welcome to the Image Authenticity Checker app! Upload an image using the sidebar to get started.")

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True, channels="RGB")

    # Predict authenticity
    with st.spinner('Analyzing the image...'):
        image = Image.open(uploaded_image)
        result = classify_image(image)

    # Display the result with styling
    st.success(f'**Result:** The image is classified as **{result}**.')

else:
    st.info("Please upload an image to analyze.")


import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to extract features from the image
def extract_features(image):
    img_array = np.array(image.resize((128, 128))) / 255.0  # Resize for efficiency
    features = []

    # Mean and standard deviation for RGB channels
    mean_rgb = img_array.mean(axis=(0, 1))
    std_rgb = img_array.std(axis=(0, 1))
    features.extend(mean_rgb)
    features.extend(std_rgb)

    # Histogram features for RGB channels
    hist_red, _ = np.histogram(img_array[:, :, 0], bins=8, range=(0, 1))
    hist_green, _ = np.histogram(img_array[:, :, 1], bins=8, range=(0, 1))
    hist_blue, _ = np.histogram(img_array[:, :, 2], bins=8, range=(0, 1))
    features.extend(hist_red[:4])
    features.extend(hist_green[:4])
    features.extend(hist_blue[:4])

    return np.array(features)

# Streamlit App Layout
st.title("Image Feature Extraction and Visualization App")
st.sidebar.title("Upload your image")

# Sidebar to upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Proceed if an image file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("Extracting features from the image...")
    with st.spinner("Processing..."):
        features = extract_features(image)

        # Define feature labels
        labels = ['Mean Red', 'Mean Green', 'Mean Blue',
                  'Std Red', 'Std Green', 'Std Blue'] + \
                 [f'Hist Red {i+1}' for i in range(4)] + \
                 [f'Hist Green {i+1}' for i in range(4)] + \
                 [f'Hist Blue {i+1}' for i in range(4)]
        
        feature_values = features.flatten()

        # Visualization 1: Bar Chart of Feature Values
        st.subheader("Bar Chart of Extracted Features")
        fig, ax = plt.subplots()
        ax.barh(labels, feature_values, color='skyblue')
        ax.set_xlabel('Feature Value')
        ax.set_title('Extracted Features from Image')
        st.pyplot(fig)

        # Visualization 2: Histogram of RGB Channel Intensities
        st.subheader("RGB Channel Histogram")
        fig, ax = plt.subplots()
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            ax.hist(np.array(image)[:, :, i].flatten(), bins=32, color=color, alpha=0.5, label=f'{color.capitalize()} Channel')
        ax.set_title('RGB Histogram')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)

        # Visualization 3: Heatmap of RGB Channels
        st.subheader("Heatmap of RGB Channels")
        img_array = np.array(image)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colormaps = ['Reds', 'Greens', 'Blues']
        for i, (color, cmap) in enumerate(zip(colors, colormaps)):
            sns.heatmap(img_array[:, :, i], ax=axes[i], cmap=cmap, cbar=False)
            axes[i].set_title(f'{color.capitalize()} Channel Intensity')
            axes[i].axis('off')
        st.pyplot(fig)

        # Visualization 4: Pairplot of RGB Channel Intensities (Sampling Pixels)
        st.subheader("Pairplot of RGB Channel Intensities")
        img_sample = img_array.reshape(-1, 3)
        max_samples = min(len(img_sample), 500)
        img_sample_df = pd.DataFrame(img_sample[np.random.choice(len(img_sample), max_samples, replace=False)], columns=['Red', 'Green', 'Blue'])
        fig = sns.pairplot(img_sample_df, plot_kws={'alpha': 0.2})
        st.pyplot(fig)

        # Visualization 5: Correlation Heatmap of Features
        st.subheader("Correlation Heatmap of Features")
        feature_df = pd.DataFrame([feature_values], columns=labels)
        corr_matrix = feature_df.corr()
        
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)

        # Download option for extracted features
        csv_data = feature_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Features as CSV", data=csv_data, file_name="extracted_features.csv", mime="text/csv")

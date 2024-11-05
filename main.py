import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract features from the image
def extract_features(image):
    """Extracts features from the image for visualization."""
    img_array = np.array(image) / 255.0  # Normalize the image
    features = []

    # Mean of RGB channels
    mean_rgb = img_array.mean(axis=(0, 1))  # Mean for R, G, B
    features.extend(mean_rgb)  # 3 features

    # Standard deviation of RGB channels
    std_rgb = img_array.std(axis=(0, 1))  # Std for R, G, B
    features.extend(std_rgb)  # 3 features

    # Extract histogram features
    hist_red, _ = np.histogram(img_array[:, :, 0], bins=8, range=(0, 1))
    hist_green, _ = np.histogram(img_array[:, :, 1], bins=8, range=(0, 1))
    hist_blue, _ = np.histogram(img_array[:, :, 2], bins=8, range=(0, 1))

    # Use the first 4 histogram bins from each channel
    features.extend(hist_red[:4])  # First 4 bins from Red
    features.extend(hist_green[:4])  # First 4 bins from Green
    features.extend(hist_blue[:4])  # First 4 bins from Blue

    return np.array(features)

# Streamlit App Layout
st.title("Image Feature Extraction and Visualization App")
st.sidebar.title("Upload your image")

# Sidebar to upload image
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Proceed if an image file is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Extract features from the image
    st.write("Extracting features from the image...")
    with st.spinner("Processing..."):
        features = extract_features(image)  # Extract features from the uploaded image

        # Debug: Print the extracted feature values
        st.write(f"Extracted Features: {features}")

        # Feature Labels for Visualization
        labels = ['Mean Red', 'Mean Green', 'Mean Blue',
                  'Std Red', 'Std Green', 'Std Blue'] + \
                 [f'Hist Red {i+1}' for i in range(4)] + \
                 [f'Hist Green {i+1}' for i in range(4)] + \
                 [f'Hist Blue {i+1}' for i in range(4)]
        feature_values = features.flatten()

        ### Visualization 1: Bar Chart of Feature Values
        st.subheader("Bar Chart of Extracted Features")
        fig, ax = plt.subplots()
        ax.barh(labels, feature_values, color='skyblue')
        ax.set_xlabel('Feature Value')
        ax.set_title('Extracted Features from Image')
        st.pyplot(fig)

        ### Visualization 2: Histogram of RGB Channel Intensities
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

        ### Visualization 3: Heatmap of RGB Channels
        st.subheader("Heatmap of RGB Channels")
        img_array = np.array(image)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colormaps = ['Reds', 'Greens', 'Blues']
        for i, (color, cmap) in enumerate(zip(colors, colormaps)):
            sns.heatmap(img_array[:, :, i], ax=axes[i], cmap=cmap, cbar=False)
            axes[i].set_title(f'{color.capitalize()} Channel Intensity')
            axes[i].axis('off')
        st.pyplot(fig)

        ### Visualization 4: Pairplot of RGB Intensities (Sampling Pixels)
        st.subheader("Pairplot of RGB Channel Intensities")
        img_sample = img_array.reshape(-1, 3)  # Flatten image pixels into RGB values
        img_sample_df = sns.load_dataset('iris').iloc[:len(img_sample), :3].copy()
        img_sample_df.columns = ['Red', 'Green', 'Blue']
        img_sample_df['Red'], img_sample_df['Green'], img_sample_df['Blue'] = img_sample[:, 0], img_sample[:, 1], img_sample[:, 2]
        
        fig = sns.pairplot(img_sample_df, plot_kws={'alpha': 0.2})
        st.pyplot(fig)

        ### Visualization 5: Correlation Heatmap of Features
        st.subheader("Correlation Heatmap of Features")
        feature_df = sns.load_dataset('iris').iloc[:1, :len(feature_values)].copy()
        feature_df.iloc[0] = feature_values
        feature_df.columns = labels
        corr_matrix = feature_df.corr()
        
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)

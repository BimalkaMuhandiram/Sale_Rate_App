# Image Feature Extraction and Visualization App

This Streamlit application allows users to upload an image and extract various features for visualization. The app computes color statistics, intensity distributions, and visual relationships between RGB channel intensities.

## Features

- **Image Upload**: Users can upload images in JPG, JPEG, or PNG formats.
- **Feature Extraction**: The application extracts and computes several features from the uploaded image, including:
  - Mean and standard deviation of RGB channels.
  - Histograms of pixel intensities for each color channel.
- **Visualizations**:
  - Bar chart of extracted features.
  - Histogram showing RGB channel intensity distributions.
  - Heatmaps for visualizing intensity variations across the RGB channels.
  - Pairplot displaying relationships between RGB channel intensities.
  - Correlation heatmap of the extracted features.
- **Downloadable CSV**: Users can download the extracted features as a CSV file.

## Requirements

To run this application, ensure you have the following Python packages installed:

- `streamlit`
- `numpy`
- `Pillow`
- `matplotlib`
- `seaborn`
- `pandas`

You can install these packages using pip:

```bash
pip install streamlit numpy Pillow matplotlib seaborn pandas

# Sale_Rate_App

This is a **Streamlit** web application that performs image classification using pre-trained models like **MobileNetV2** and **VGG16**. The app allows users to upload images, enhance them (adjust brightness and contrast), and classify the images to identify objects within them. Multiple graphs are displayed to visualize the classification results, including bar charts, pie charts, line charts, histograms, and scatter plots.

## Features

1. **Pre-trained Model Selection**:
    - Choose between two pre-trained models: **MobileNetV2** and **VGG16**.
    - Models are pre-trained on ImageNet and can classify a wide range of objects.

2. **Image Upload and Enhancements**:
    - Users can upload `.jpg`, `.jpeg`, and `.png` image files.
    - Adjust **brightness** and **contrast** of the image using sliders.

3. **Top-N Predictions**:
    - Users can choose the number of top predictions to display (between 1 and 10).

4. **Classification Results**:
    - The app shows the top predictions and confidence scores.
    - Progress bar updates during classification to indicate processing status.

5. **Graphs and Visualizations**:
    - **Bar Chart**: Visualize confidence scores of top predictions.
    - **Pie Chart**: Display proportions of the confidence scores.
    - **Line Chart**: See the trend of confidence scores across predictions.
    - **Histogram**: View the frequency distribution of confidence scores.
    - **Scatter Plot**: Visualize the distribution of confidence scores across labels.

## How to Run the Application

### Local Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/BimalkaMuhandiram/Sale_Rate_App.git
    cd streamlit-image-classification-app
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```

4. Open your browser and navigate to the provided local URL (e.g., `http://localhost:8501`).

### Using the App

1. Upload an image by selecting a `.jpg`, `.jpeg`, or `.png` file.
2. Adjust the **brightness** and **contrast** using the sliders in the sidebar.
3. Choose a pre-trained model (`MobileNetV2` or `VGG16`).
4. Select the number of top predictions to display using the **Top-N** slider.
5. The results, including classification predictions and their confidence scores, will be displayed with various graphs for visualization.

## App Features (Screenshots)

### Main Screen
![Main Screen](path_to_your_image)

### Sidebar
![Sidebar](path_to_your_image)

### Image Upload and Enhancements
![Image Upload](path_to_your_image)

### Graphs
- **Bar Chart**: Displays the confidence scores of top predictions.
![Bar Chart](path_to_your_image)

- **Pie Chart**: Shows the proportion of each prediction's confidence.
![Pie Chart](path_to_your_image)

- **Line Chart**: Trends of confidence scores.
![Line Chart](path_to_your_image)

- **Histogram**: Frequency of confidence scores.
![Histogram](path_to_your_image)

- **Scatter Plot**: Distribution of confidence scores.
![Scatter Plot](path_to_your_image)

## Model Information

1. **MobileNetV2**:
    - Lightweight model optimized for mobile devices, trained on the ImageNet dataset.
    
2. **VGG16**:
    - Larger model with a deeper architecture, trained on the ImageNet dataset, known for its strong performance on image classification tasks.

## Deployment

This app can be deployed on **Streamlit Cloud** by following these steps:

1. Push your project to GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect your GitHub repository and select the branch containing your app code.
4. Click **Deploy** and your app will be live!

## Dependencies

- **Python 3.8+**
- **TensorFlow** (for loading pre-trained models)
- **Streamlit** (for creating the web interface)
- **Pillow** (for image processing)
- **Matplotlib** (for data visualization)

Install all dependencies using:
```bash
pip install -r requirements.txt

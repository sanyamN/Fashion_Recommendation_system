# Fashion Recommendation System

This project implements a Fashion Recommendation System that identifies and recommends visually similar images from a dataset. It utilizes deep learning and machine learning techniques for feature extraction and similarity computation.

## Overview

The system is built using a pre-trained convolutional neural network (CNN) model, `ResNet50`, to extract high-level features from images. These features are then used to find similar images based on their visual characteristics. A k-Nearest Neighbors (k-NN) model is employed for the similarity search.

---
## Dataset

The dataset used for this project contains a collection of fashion product images. It can be downloaded from the following link:  
[Download Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

---

## Workflow

### 1. Data Preparation
All the images that form the basis of recommendations are stored in a designated directory. The paths of these images are collected and organized into a list for further processing.

### 2. Feature Extraction
A pre-trained ResNet50 model, with its top classification layer removed, is used to extract feature vectors from the images. The feature extraction process includes resizing the images, preprocessing them to match the ResNet50 input requirements, and normalizing the resulting feature vectors for uniformity.

### 3. Feature Vector Storage
The extracted feature vectors and the corresponding image filenames are serialized and saved into files. This enables efficient reuse of the computed features without requiring recomputation for every run.

### 4. Building the Recommendation Model
A k-NN model is trained using the feature vectors. The model computes the similarity between images by measuring the Euclidean distance between their feature vectors. This distance metric helps identify the most visually similar images.

### 5. Generating Recommendations
When a user uploads an image, its features are extracted using the same ResNet50-based process. The k-NN model then identifies the closest matches to the uploaded image from the dataset, and their corresponding images are retrieved as recommendations.

---

## Running the Application
The application is implemented as a Streamlit web app. Users can upload an image, and the system will display the uploaded image alongside its top recommendations.

Here is an example of how the application appears:

![SS of Web app](https://github.com/user-attachments/assets/8d81e1c3-8464-45e2-b1bc-f4421bc7962e)

### Steps:
1. Start the application using Streamlit.
2. Upload an image via the web interface.
3. View the uploaded image and its visually similar recommendations displayed below it.

---

## Key Technologies Used

1. **ResNet50**: A pre-trained deep learning model used for feature extraction.
2. **k-Nearest Neighbors (k-NN)**: A machine learning algorithm used for similarity-based image retrieval.
3. **Streamlit**: A Python library used to build the interactive web application.
4. **Pandas and NumPy**: For data handling and numerical computations.
5. **Pickle**: For saving and loading precomputed data, such as feature vectors and filenames.

---

## Use Cases
- Fashion e-commerce platforms to recommend similar products.
- Content-based image retrieval systems.
- Style matching or outfit pairing applications.

---

## Future Enhancements
1. Integration of more advanced nearest neighbor search algorithms for faster recommendations on larger datasets.
2. Expansion of the recommendation criteria to include additional metadata, such as color or style.
3. Development of a more sophisticated user interface with functionalities like multi-image uploads and filters.

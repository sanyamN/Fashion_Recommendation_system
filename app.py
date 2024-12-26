import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import streamlit as st

# Load precomputed image features and filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Define the function to extract features from an uploaded image
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# Load the pre-trained ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([
    model,
    GlobalMaxPool2D()
])

# Configure the Nearest Neighbors model for recommendations
neighbors = NearestNeighbors(n_neighbors=8, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Ensure the 'upload' directory exists
if not os.path.exists('upload'):
    os.makedirs('upload')

# Streamlit app starts here
st.title("Fashion Recommendation System")
upload_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if upload_file is not None:
    # Save the uploaded file to the 'upload' directory
    file_path = os.path.join('upload', upload_file.name)
    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    st.subheader("Uploaded Image")
    st.image(file_path)

    # Extract features and generate recommendations
    input_img_features = extract_features_from_images(file_path, model)
    distance, indices = neighbors.kneighbors([input_img_features])

    st.subheader('Recommended Images')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(filenames[indices[0][1]])
    with col2:
        st.image(filenames[indices[0][2]])
    with col3:
        st.image(filenames[indices[0][3]])
    with col4:
        st.image(filenames[indices[0][4]])
    with col5:
        st.image(filenames[indices[0][5]])

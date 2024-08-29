import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU
import imageio

# Define custom GRU to handle unsupported arguments
class CustomGRU(GRU):
    def __init__(self, **kwargs):
        kwargs.pop('time_major', None)  # Remove unsupported argument
        super().__init__(**kwargs)

# Load the pre-trained model
try:
    model = load_model('rnn_model.h5', custom_objects={"GRU": CustomGRU})
except ValueError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Paths to your folders
DATA_FOLDER = 'C:/Users/101ri/OneDrive/Desktop/New DeepFake/Video Folder'
TEST_FOLDER = 'C:/Users/101ri/OneDrive/Desktop/New DeepFake/test_videos'

# Constants
MAX_SEQ_LENGTH = 20  # Adjust according to your actual sequence length
NUM_FEATURES = 2048  # Adjust according to the feature extractor output dimensions

# Placeholder for feature extractor
feature_extractor = tf.keras.applications.Xception(weights="imagenet", include_top=False, pooling="avg")

def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(224, 224)):
    cap = imageio.get_reader(path, 'ffmpeg')
    frames = []
    for i, frame in enumerate(cap):
        if len(frames) < max_frames:
            frame = tf.image.resize(frame, resize)
            frames.append(frame.numpy())
        else:
            break
    cap.close()
    return np.array(frames)

def prepare_single_video(frames):
    frames = frames[None, ...]  # Add batch dimension
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[1]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[j, :, :][None, ...])
        frame_mask[i, :length] = 1

    return frame_features, frame_mask

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH, 224, 224, 3)),
        GRU(128, return_sequences=True),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

def sequence_prediction(video_path):
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames)
    adjusted_model = build_model()
    adjusted_model.load_weights('rnn_model.h5')
    return adjusted_model.predict([frame_features, frame_mask])[0]

# Streamlit app
st.title("Deep Fake Video Detection")

# File uploader widget
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    video_path = os.path.join(DATA_FOLDER, TEST_FOLDER, uploaded_file.name)
    
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Prediction
    try:
        prediction = sequence_prediction(video_path)
        if prediction >= 0.5:
            st.error("The video is predicted to be FAKE.")
        else:
            st.success("The video is predicted to be REAL.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

    # Play video
    st.video(uploaded_file)

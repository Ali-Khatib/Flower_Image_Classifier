import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pathlib

# Function to process the image
def process_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array /= 255.0
    return np.expand_dims(img_array, axis=0)

# Prediction function
def predict(img_path, model, top_k=5, class_names=None):
    processed_img = process_image(img_path)
    predictions = model.predict(processed_img)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_probs = predictions[top_indices]

    # Map indices to class labels
    class_indices = [str(i) for i in top_indices]
    if class_names:
        mapped_classes = [class_names.get(str(i), f"Class {i}") for i in top_indices]
    else:
        mapped_classes = class_indices

    # Print results
    print("\nTop K Predictions:")
    for i in range(top_k):
        print(f"{mapped_classes[i]}: {top_probs[i]:.4f}")

    return mapped_classes[0], top_probs[0]  # return most likely prediction and prob


# Argument parser
def main():
    parser = argparse.ArgumentParser(description='Flower image classifier')
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('model_path', type=str, help='Path to trained Keras model')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping labels to flower names')

    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path, compile=False)

    # Load category names if provided
    class_names = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)

    # Run prediction
    predict(args.image_path, model, args.top_k, class_names)

if __name__ == '__main__':
    main()

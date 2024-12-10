import cv2
import joblib
import numpy as np
import requests
from io import BytesIO
from PIL import Image

# Load the trained model
model_path = "npk_ph_predictor_model.pkl"
model = joblib.load(model_path)

# Feature extraction function for a single image
def extract_features_for_single_image(image):
    # Resize the image
    image = cv2.resize(image, (224, 224))

    # Compute mean and std for RGB channels
    mean_color = np.mean(image, axis=(0, 1))
    std_color = np.std(image, axis=(0, 1))

    # Compute normalized RGB
    sum_rgb = np.sum(mean_color)
    norm_r = mean_color[2] / sum_rgb
    norm_g = mean_color[1] / sum_rgb
    norm_b = mean_color[0] / sum_rgb

    # Nitrogen indicator (G dominance)
    green_dominance = mean_color[1] > (mean_color[0] + mean_color[2]) / 2

    # Phosphorus indicator (Bluish soil)
    blue_ratio = mean_color[0] / (mean_color[1] + mean_color[2])

    # Potassium: Yellowish-Blue Ratio
    yellowish_blue_ratio = ((mean_color[2] + mean_color[1]) / 2) / mean_color[0]

    # Texture Features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Local Binary Pattern (LBP)
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)

    # GLCM (Gray Level Co-occurrence Matrix)
    from skimage.feature import graycomatrix, graycoprops
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, "contrast")[0, 0]
    glcm_homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    glcm_entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))

    # Combine features into a flat dictionary
    features = {
        "mean_R": mean_color[2],
        "mean_G": mean_color[1],
        "mean_B": mean_color[0],
        "std_R": std_color[2],
        "std_G": std_color[1],
        "std_B": std_color[0],
        "norm_R": norm_r,
        "norm_G": norm_g,
        "norm_B": norm_b,
        "green_dominance": green_dominance,
        "blue_ratio": blue_ratio,
        "yellowish_blue_ratio": yellowish_blue_ratio,
        "lbp_hist": lbp_hist.tolist(),
        "glcm_contrast": glcm_contrast,
        "glcm_homogeneity": glcm_homogeneity,
        "glcm_entropy": glcm_entropy,
    }

    # Return features
    return features

# Function to read image from URL
def read_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    # Convert image to OpenCV format (BGR)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

# Predict NPK and pH for a single image from a URL
def predict_for_url_image(image_url):
    # Read image from URL
    image = read_image_from_url(image_url)

    # Extract features
    features = extract_features_for_single_image(image)

    # Prepare features for prediction
    X_test = {
        key: [value] if not isinstance(value, list) else value
        for key, value in features.items()
    }

    # Drop unused features (e.g., lbp_hist if not used in training)
    if "lbp_hist" in X_test:
        del X_test["lbp_hist"]

    # Convert to NumPy array for prediction
    X_test = np.array(list(X_test.values())).T

    # Predict using the loaded model
    predictions = model.predict(X_test)[0]
    results = {
        "Nitrogen(%)": predictions[0],
        "Phosphorus(ppm)": predictions[1],
        "Potassium(ppm)": predictions[2],
        "pH": predictions[3],
    }

    return results

# Test with an image URL
image_url = "https://th.bing.com/th/id/OIP.2JAh4BRCX_xT73yZD0Vp1gHaEL?w=249&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7"  # Replace with your image URL
results = predict_for_url_image(image_url)

# Print the results
print(f"Predictions for image at URL '{image_url}':")
print(f"Nitrogen (%): {results['Nitrogen(%)']:.2f}")
print(f"Phosphorus (ppm): {results['Phosphorus(ppm)']:.2f}")
print(f"Potassium (ppm): {results['Potassium(ppm)']:.2f}")
print(f"pH: {results['pH']:.2f}")

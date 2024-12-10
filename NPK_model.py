import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # To save and load the model

# Step 1: Feature Extraction for NPK and pH
def extract_features(image_path):
    image = cv2.imread(image_path)
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
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)

    # GLCM (Gray Level Co-occurrence Matrix)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, "contrast")[0, 0]
    glcm_homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    glcm_entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))

    # Combine features
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
    return features

# Step 2: Map Features to NPK and pH Values
def predict_npk_and_ph(features):
    # Nitrogen Prediction
    green_ratio = features["mean_G"] / ((features["mean_R"] + features["mean_B"]) + features["mean_G"])
    if green_ratio > 0.6:
        nitrogen = 0.05 + (green_ratio - 0.6) * (0.07 - 0.05) / (1 - 0.6)
    elif green_ratio > 0.33:
        nitrogen = 0.02 + (green_ratio - 0.33) * (0.05 - 0.02) / (0.6 - 0.33)
    else:
        nitrogen = 0.01 + (0.02 / 0.33) * green_ratio

    # Phosphorus Prediction
    if features["blue_ratio"] > 0.8:
        phosphorus = 20 + (features["blue_ratio"] - 0.8) * (25 - 20) / (1 - 0.8)
    elif features["blue_ratio"] > 0.3:
        phosphorus = 6 + (features["blue_ratio"] - 0.3) * (20 - 6) / (0.8 - 0.3)
    else:
        phosphorus = 1 + (5 / 0.3) * features["blue_ratio"]

    # Potassium Prediction
    if features["yellowish_blue_ratio"] > 1.4:
        potassium = 150 + (features["yellowish_blue_ratio"] - 1.4) * (300 - 150) / (2 - 1.4)
        potassium = min(potassium, 300)
    elif features["yellowish_blue_ratio"] > 1.0:
        potassium = 50 + (features["yellowish_blue_ratio"] - 1.0) * (150 - 50) / (1.4 - 1.0)
    else:
        potassium = 10 + (40 / 1.0) * features["yellowish_blue_ratio"]

    # pH Prediction
    mean_brightness = (features["mean_R"] + features["mean_G"] + features["mean_B"]) / 3
    if features["blue_ratio"] > 0.6 and mean_brightness < 80:
        ph = 4.5 + (0.6 - features["blue_ratio"]) * 2
    elif 80 <= mean_brightness <= 150:
        ph = 6.5 + (features["yellowish_blue_ratio"] - 1.0) * 1.5
    else:
        ph = 8.0 + (features["mean_R"] / features["mean_B"]) * 0.5
    ph = min(max(ph, 3.5), 9.0)  # Clamp pH to realistic values

    return {"Nitrogen(%)": nitrogen, "Phosphorus(ppm)": phosphorus, "Potassium(ppm)": potassium, "pH": ph}

# Step 3: Process Dataset
def process_dataset(folder_path):
    feature_data = []

    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if file.endswith((".jpg", ".png", ".jpeg")):
                    # Extract features
                    features = extract_features(file_path)
                    npk_ph_values = predict_npk_and_ph(features)
                    features.update(npk_ph_values)  # Add NPK and pH to feature vector
                    features["soil_type"] = subfolder  # Add soil type as label
                    feature_data.append(features)
    return pd.DataFrame(feature_data)

# Step 4: Train a Model (Optional)
def train_model(train_df):
    # Prepare feature matrix and target labels
    X = train_df.drop(columns=["Nitrogen(%)", "Phosphorus(ppm)", "Potassium(ppm)", "pH", "soil_type", "lbp_hist"])
    y = train_df[["Nitrogen(%)", "Phosphorus(ppm)", "Potassium(ppm)", "pH"]]

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Validate the model
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    print("Validation MSE:", mse)

    return model

# Step 5: Automate the Process
if __name__ == "__main__":
    train_dir = "Final/Normal/Trains"
    test_dir = "Final/Normal/Tests"

    # Process training and testing datasets
    train_df = process_dataset(train_dir)
    test_df = process_dataset(test_dir)

    # Save processed data
    train_df.to_csv("train_features_with_pH.csv", index=False)
    test_df.to_csv("test_features_with_pH.csv", index=False)

    # Train and validate the model
    print("Training the model...")
    model = train_model(train_df)

    # Save the model
    joblib.dump(model, "npk_ph_predictor_model.pkl")
    print("Model saved as 'npk_ph_predictor_model.pkl'")

    print("Feature extraction, pH prediction, and model training completed!")

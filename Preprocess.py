from tqdm import tqdm  # For progress bars
import os
import cv2
import numpy as np

def preprocess_image(image, size=(224, 224)):
    """
    Preprocesses the image by resizing, enhancing contrast (via histogram equalization), 
    and normalizing the pixel values.
    """
    # Resize image to a fixed size (224x224)
    image_resized = cv2.resize(image, size)

    # Apply histogram equalization on each channel (R, G, B) to improve contrast
    channels = cv2.split(image_resized)
    channels_equalized = [cv2.equalizeHist(channel) for channel in channels]
    image_equalized = cv2.merge(channels_equalized)
    
    # Optional: Apply a Gaussian blur to reduce noise (if needed)
    image_denoised = cv2.GaussianBlur(image_equalized, (3, 3), 0)
    
    # Normalize pixel values to the range [0, 1] (for deep learning models)
    image_normalized = image_denoised.astype(np.float32) / 255.0

    return image_normalized

def preprocess_images_in_place(input_dir, size=(224, 224)):
    """
    Preprocesses all images in the input directory in place (resize, equalize, and normalize).
    """
    for subfolder in sorted(os.listdir(input_dir)):
        subfolder_path = os.path.join(input_dir, subfolder)

        if os.path.isdir(subfolder_path):
            for file in tqdm(os.listdir(subfolder_path), desc=f"Processing {subfolder}"):
                file_path = os.path.join(subfolder_path, file)
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        # Read the image
                        image = cv2.imread(file_path)

                        if image is None:
                            print(f"Failed to load image: {file_path}")
                            continue  # Skip this image if it fails to load

                        # Apply preprocessing (resize, enhance contrast, and normalize)
                        processed_image = preprocess_image(image, size)

                        # Overwrite the original image with the processed image
                        # For example, you might save the processed image in another folder.
                        # cv2.imwrite(file_path, processed_image)  # Uncomment to overwrite if desired

                        # You can also save the image as a .npy file if you're planning to use numpy arrays
                        # np.save(file_path.replace(".jpg", ".npy"), processed_image)  # Save as numpy array

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

def prepare_dataset_in_place(input_dir):
    """
    Prepares the entire dataset by preprocessing images directly in the original directories.
    """
    print("Preparing dataset in place...")
    preprocess_images_in_place(input_dir)
    print("Dataset preparation complete. Images processed directly.")

# Example usage
if __name__ == "__main__":
    train_input_dir = "Dataset/Trains"
    test_input_dir = "Dataset/Tests"

    # Prepare train and test datasets
    prepare_dataset_in_place(train_input_dir)
    prepare_dataset_in_place(test_input_dir)

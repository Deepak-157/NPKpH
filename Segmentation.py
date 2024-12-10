from tqdm import tqdm  # For progress bars
import os
import cv2
import numpy as np

def segment_image(image, size=(224, 224)):
    """
    Segments the soil from the background using Otsu's thresholding method and resizes the image.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optionally, apply morphological transformations to clean up the segmentation
    kernel = np.ones((5, 5), np.uint8)  # Kernel size can be adjusted
    segmented = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)  # Close small gaps
    
    # Convert back to BGR format if you want to keep the color channels intact
    segmented_colored = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)

    # Resize the segmented image to the specified size
    resized_image = cv2.resize(segmented_colored, size)

    return resized_image

def segment_and_resize_images_in_place(input_dir, size=(224, 224)):
    """
    Segments and resizes all images in the input directory in place.
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

                        # Apply segmentation and resizing (soil segmentation from background)
                        processed_image = segment_image(image, size)

                        # Save the processed (segmented and resized) image
                        cv2.imwrite(file_path, processed_image)  # Overwrite the original image with processed one

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

def prepare_dataset_for_segmentation_and_resizing(input_dir, size=(224, 224)):
    """
    Prepares the entire dataset by segmenting and resizing images directly in the original directories.
    """
    print("Segmenting and resizing dataset in place...")
    segment_and_resize_images_in_place(input_dir, size)
    print("Dataset segmentation and resizing complete. Images processed directly.")

# Example usage
if __name__ == "__main__":
    train_input_dir = "Dataset/Trains"
    test_input_dir = "Dataset/Tests"

    # Prepare train and test datasets
    prepare_dataset_for_segmentation_and_resizing(train_input_dir)
    prepare_dataset_for_segmentation_and_resizing(test_input_dir)

import os
from PIL import Image
import imagehash

# Function to calculate the hash of an image
def calculate_image_hash(image_path):
    try:
        img = Image.open(image_path)
        # Using perceptual hash to calculate a hash
        hash_value = imagehash.phash(img)
        return hash_value
    except Exception:
        return None

# Function to remove duplicate images and rename them
def remove_duplicate_images_and_rename(folder_path):
    seen_hashes = {}  # Dictionary to store unique image hashes
    index = 0  # Initialize index for renaming images

    # Iterate over the subfolders and images
    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it's a folder
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if file.endswith((".jpg", ".png", ".jpeg")):  # Valid image file
                    image_hash = calculate_image_hash(file_path)
                    if image_hash:
                        # If hash already exists, it's a duplicate
                        if image_hash in seen_hashes:
                            os.remove(file_path)  # Delete duplicate image
                        else:
                            # Rename the image to parentfolder_index
                            new_name = f"{subfolder}_{index}.jpg"  # Change extension as needed
                            new_file_path = os.path.join(subfolder_path, new_name)
                            os.rename(file_path, new_file_path)
                            seen_hashes[image_hash] = new_file_path
                            index += 1  # Increment index for the next image

# Set dataset paths
train_dir = "Final/Segmented/Trains"
test_dir = "Final/Segmented/Tests"

# Remove duplicates and rename images in both train and test folders
remove_duplicate_images_and_rename(train_dir)
remove_duplicate_images_and_rename(test_dir)

print("Duplicate removal and renaming complete!")

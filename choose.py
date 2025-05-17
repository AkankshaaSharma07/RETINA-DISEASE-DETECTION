from PIL import Image

import os
import random
import shutil

# # Define paths
# training_masked_images_path = 'retina-collected-dataset/training/masked_images'
# training_retinal_images_path = 'retina-collected-dataset/training/retinal_images'
# test_masked_images_path = 'retina-collected-dataset/test/masked_images'
# test_retinal_images_path = 'retina-collected-dataset/test/retinal_images'

# # Create test directories if they don't exist
# os.makedirs(test_masked_images_path, exist_ok=True)
# os.makedirs(test_retinal_images_path, exist_ok=True)

# # Get list of image names in both directories
# masked_images = os.listdir(training_masked_images_path)
# retinal_images = os.listdir(training_retinal_images_path)

# # Ensure both directories have the same images
# common_images = set(masked_images).intersection(retinal_images)

# # Ensure there are at least 40 common images
# if len(common_images) < 40:
#     raise ValueError("Not enough common images to select 40.")

# # Randomly select 40 images
# selected_images = random.sample(list(common_images), 130)

# # Copy selected images to test directories
# for image in selected_images:
#     shutil.move(os.path.join(training_masked_images_path, image), os.path.join(test_masked_images_path, image))
#     shutil.move(os.path.join(training_retinal_images_path, image), os.path.join(test_retinal_images_path, image))

# print(f"Successfully copied {len(selected_images)} images to the test directories.")


# Define paths
combined_data_path = 'CombinedData'

# Get list of image names in the combined data directory
combined_images = []
for root, dirs, files in os.walk(combined_data_path):
    for file in files:
        combined_images.append(os.path.relpath(os.path.join(root, file), combined_data_path))

# Convert images to PNG if not already in PNG format
for image_name in combined_images:
    if not image_name.lower().endswith('.png'):
        image_path = os.path.join(combined_data_path, image_name)
        image = Image.open(image_path)
        new_image_name = os.path.splitext(image_name)[0] + '.png'
        new_image_path = os.path.join(combined_data_path, new_image_name)
        image.save(new_image_path, 'PNG')
        os.remove(image_path)

print("Successfully converted all non-PNG images to PNG format.")
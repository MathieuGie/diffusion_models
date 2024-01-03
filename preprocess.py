from PIL import Image
import os

def resize_images(folder, output_folder, size=(256, 256)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder):
        # Check if the file is a JPEG image
        if filename.endswith('.jpg') and not filename.startswith('.'):
            img = Image.open(os.path.join(folder, filename))
            # Resize the image and use ANTIALIAS filter to maintain quality
            img = img.resize(size)
            # Save the resized image to the output folder
            img.save(os.path.join(output_folder, filename))

# Define the path to the folder containing the images
source_folder = 'G:/Mon Drive/Polytechnique_M2/Deep_Learning/Dataset/animal'
# Define the path to the folder where resized images will be saved
destination_folder = 'G:/Mon Drive/Polytechnique_M2/Deep_Learning/Dataset/resize_animal'

# Call the function with the updated folder paths
resize_images(source_folder, destination_folder)



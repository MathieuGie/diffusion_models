from PIL import Image
import os

def resize_images(folder, output_folder, size=(200, 200)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    id = 0
    for filename in os.listdir(folder):
        # Check if the file is a JPEG image
        if filename.endswith('.jpg') and not filename.startswith('.'):
            img = Image.open(os.path.join(folder, filename))
            # Resize the image and use ANTIALIAS filter to maintain quality
            img = img.resize(size)

            print(img.size)
            # Save the resized image to the output folder
            img.save(os.path.join(output_folder, "image_"+str(id)+".jpg"))
            id+=1

# Define the path to the folder containing the images
source_folder = os.getcwd() + '/dataset/CAT_00'
# Define the path to the folder where resized images will be saved
destination_folder = os.getcwd() + '/dataset/CAT_00_treated'

# Call the function with the updated folder paths
resize_images(source_folder, destination_folder)




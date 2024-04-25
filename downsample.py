from PIL import Image
import os

def resize_images(folder_path, output_folder=None, size=(512, 512)):
    """
    Resize all PNG images in the specified folder to the given size.

    Args:
    folder_path (str): The path to the folder containing the images.
    output_folder (str): The path to the folder where resized images will be saved.
                        If None, images will be resized in place.
    size (tuple): The target size of the images as (width, height).
    """
    # Create the output folder if it does not exist and if specified
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the folder_path
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img = img.resize(size, Image.Resampling.LANCZOS)

            # Save the image
            if output_folder:
                save_path = os.path.join(output_folder, filename)
            else:
                save_path = img_path
            
            img.save(save_path)

            print(f'Resized and saved: {save_path}')

source_folder = 'DIV2K_train_HR/'
output_folder = 'DIV2K_train_512/'
resize_images(source_folder, output_folder)

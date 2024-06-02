import os
import glob
from PIL import Image
from rembg import new_session, remove

# For handling CR2 raw images
try:
    import rawpy
    import imageio

    CR2_SUPPORT = True
except ImportError:
    CR2_SUPPORT = False
    print("rawpy or imageio not installed. CR2 files will be skipped.")


def process_image(input_path, output_path, session):
    if input_path.lower().endswith('.cr2') and CR2_SUPPORT:
        # Special handling for CR2 files
        with rawpy.imread(input_path) as raw:
            rgb = raw.postprocess()
        img = Image.fromarray(rgb)
    else:
        # For standard formats
        img = Image.open(input_path)

    output = remove(img, session=session)

    # Check if the output image has an alpha channel
    if output.mode == 'RGBA':
        # Get the alpha channel
        alpha = output.getchannel('A')
        # Process the alpha channel: set semi-transparent pixels (alpha < 128) to fully transparent (0)
        processed_alpha = alpha.point(lambda p: 0 if p < 128 else 255)
        # Update the image with the processed alpha channel
        output.putalpha(processed_alpha)

    output = output.crop(output.getbbox())  # Trim the empty space from removing the background

    output.save(output_path, 'PNG', quality=100, optimize=True)


def crawl_directory(input_directory, output_directory, session):
    # List of image extensions to look for
    extensions = ['*.png', '*.jpeg', '*.jpg', '*.cr2', '*.tiff']

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for ext in extensions:
        for input_path in glob.glob(os.path.join(input_directory, ext)):
            filename = os.path.basename(input_path)
            # Change file extension to .png for output
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_directory, output_filename)
            print(f"Processing {input_path} -> {output_path}")
            process_image(input_path, output_path, session)


# Define your input and output directories
input_directory = 'Canon/'
output_directory = 'Canon/u2net'

model_name = "u2net"  # MCOnet failing with new setup
session = new_session(model_name)

# Crawl the directory and process all images
crawl_directory(input_directory, output_directory, session)

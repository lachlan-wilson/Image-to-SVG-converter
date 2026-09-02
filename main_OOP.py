# ---- Image2Vector ----
#  ---- 02/08/2026 ----
import os
from pathlib import Path
import tempfile
import subprocess
from PIL import Image, ImageOps
import cv2
import numpy
from sklearn.cluster import MiniBatchKMeans
import time
import shutil
potrace_path = "/usr/local/bin/potrace"     # Absolute path of Potrace


def title(title, sub=False):
    """
    Turn a string into a title with a constant width.

    Parameters
    ----------
    title: str
        The string to be formatted.
    sub: bool, default=False
        Whether the string is a subtitle.

    Returns
    -------
    str
        The formatted string.
    """
    char_length = int(
        (30 if sub else 50 - len(title)) / 2)  # Make a constant width no matter the length of the title
    offset = 1 if len(title) % 2 == 1 else 0  # Account for half-spaces

    # Display the title in blue with correct length
    colour = "\033[32m" if sub else "\033[94m"
    return colour + "<" + "-" * char_length + f" {title} " + "-" * (char_length + offset) + ">\033[0m"


class MyImage:
    def __init__(self, defaults=("test_image.jpg", 8, 50, 100, 30, 1), image_types=(".jpeg", ".jpg", ".png")):
        defaults = {"image_name": defaults[0],
                    "colour_depth": defaults[1],
                    "min_contour_area": defaults[2],
                    "max_contour_distance": defaults[3],
                    "bridge_width": defaults[4],}

        # User input functions
        def _get_image_path():
            """
            Get a valid image name from the user and make it a `Path` object for an image in the `images` folder.
            The file must be an image of a supported type and exist within the `images` folder.

            Returns
            -------
            str
                A string containing the path to the image, e.g. `images/test_image.jpg`.
            """
            # Loop until a valid image name is entered
            while True:
                # Get the image name from the user
                image_name = str(input(f"Image name [{defaults['image_name']}]: ") or defaults["image_name"])

                # Ensure the image name is not a path
                if "/" in image_name:
                    print(f"\033[91mError. Invalid image name, cannot be a path. Please enter a valid file name.\033[0m")
                    continue

                # Convert the image name to a Path within the `images` folder
                image_name = Path("images") / image_name

                # Ensure the image name is a valid file type
                if image_name.suffix not in image_types:
                    print(f"\033[91mError. Invalid file type. Please enter a valid file type {image_types}.\033[0m")
                    continue

                # Ensure the image name exists within the `images` folder
                if not image_name.is_file():
                    print(f"\033[91mError. File not found. Please enter a valid file name.\033[0m")
                    continue

                return image_name

        def _get_colour_depth():
            """
            Get a valid colour depth from the user.
            The colour depth must be an integer greater than 0, and it is recommended to be less than 51.

            Returns
            -------
            int
                An integer representing the colour depth.
            """
            while True:
                # Get the colour depth from the user ensuring it is an integer
                try:
                    colour_depth = int(input(f"Number of colours [{defaults['colour_depth']}]: ") or defaults["colour_depth"])

                except ValueError:
                    print(f"\033[91mError. Invalid data type. Please enter an integer.\033[0m")
                    continue

                if not colour_depth > 0:
                    print(f"\033[91mError. Invalid integer ({colour_depth}). Please enter an integer greater than 0.\033[0m")
                    continue

                # Warn the user if the colour depth is > 50
                if colour_depth > 50:
                    while True:
                        answer = input(f"\033[33mWarning. Large integer. Colour depths greater than 50 are not recommended, are you sure you want to proceed (Y/[N])? \033[0m") or "N"
                        if answer.upper() not in ("Y", "N"):
                            print(f"\033[91mError. Invalid input. Please enter 'Y' or 'N'.\033[0m")
                            continue
                        break
                    if answer.upper() == "N":
                        continue

                return colour_depth

        def _get_min_contour_area():
            """
            Get a valid minimum contour area from the user.
            The minimum contour area must be an integer greater than 0.

            Returns
            -------
            int
                An integer representing the minimum contour area.
            """
            while True:
                try:
                    min_contour_area = int(input(f"Minimum contour area (px\u00b2) [{defaults['min_contour_area']}]: ") or defaults['min_contour_area'])

                except ValueError:
                    print(f"\033[91mError. Invalid data type. Please enter an integer.\033[0m")
                    continue

                if not min_contour_area > 0:
                    print(f"\033[91mError. Invalid integer ({min_contour_area}). Please enter an integer greater than 0.\033[0m")
                    continue

                return min_contour_area

        def _get_max_contour_distance():
            """
            Get a valid maximum contour distance from the user.
            The maximum contour distance must be an integer greater than 0.

            Returns
            -------
            int
                An integer representing the maximum contour distance.
            """
            while True:
                # TODO: Change from center to centre
                try:
                    max_contour_distance = int(input(f"Maximum contour distance (center to center) (px) [{defaults['max_contour_distance']}]: ") or defaults["max_contour_distance"])

                except ValueError:
                    print(f"\033[91mError. Invalid data type. Please enter an integer.\033[0m")
                    continue

                if not max_contour_distance > 0:
                    print(f"\033[91mError. Invalid integer ({max_contour_distance}). Please enter an integer greater than 0.\033[0m")
                    continue

                return max_contour_distance

        def _get_bridge_width():
            """
            Get a valid bridge width from the user.
            The bridge width must be an integer greater than 0, and it is recommended to be less than 51.

            Returns
            -------
            int
                An integer representing the bridge width.
            """
            while True:
                try:
                    bridge_width = int(input(f"Bridge Width (px) [{defaults['bridge_width']}]: ") or defaults["bridge_width"])

                except ValueError:
                    print(f"\033[91mError. Invalid data type. Please enter an integer.\033[0m")
                    continue

                if not bridge_width > 0:
                    print(f"\033[91mError. Invalid integer ({bridge_width}). Please enter an integer greater than 0.\033[0m")
                    continue

                if bridge_width > 50:
                    while True:
                        answer = input(
                            f"\033[33mWarning. Large integer. Bridge widths greater than 50 are not recommended, are you sure you want to proceed (Y/[N])? \033[0m") or "N"
                        if answer.upper() not in ("Y", "N"):
                            print(f"\033[91mError. Invalid input. Please enter 'Y' or 'N'.\033[0m")
                            continue
                        break
                    if answer.upper() == "N":
                        continue

                return bridge_width

        # Other functions
        def _get_inputs():
            """
            Get all the validated input parameters from the user.

            Returns
            -------
            tuple[str, int, int, int, int]
                A tuple containing the validated input parameters.
            """
            print(title("Input Parameters"))

            return _get_image_path(), _get_colour_depth(), _get_min_contour_area(), _get_max_contour_distance(), _get_bridge_width()

        def _load_image(img_path):
            """
            Load the image from the given path into a numpy array.
            The image is correctly orientated in RGBA format.

            Parameters
            ----------
            img_path : `Path`
                The path to the image within the `images` folder.

            Returns
            -------
            ImageFile
                A numpy array containing the image pixels.
            """
            print(title("Loading Image"))

            print("Loading image...", end="")
            img = Image.open(img_path)  # Open the image
            img = ImageOps.exif_transpose(img)  # Ensure image is correctly orientated
            img = img.convert("RGBA")   # Convert the image to RGBA
            img = numpy.array(img)  # Convert the image to a numpy array of pixels
            print("\rLoaded Image.\n")

            return img


image = MyImage()

# ---- Image2Vector ----
#  ---- 02/08/2026 ----
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import time
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

POTRACE_PATH = "/usr/local/bin/potrace"  # Absolute path of Potrace


def title(title_string: str, sub: bool = False) -> str:
    """
    Format a title or subtitle with dashes and ANSI colour codes.

    Parameters
    ----------
    title_string : str
        Text to format, excluding any decoration.
    sub : bool, optional
        Whether to use shorter, green subtitle styling instead of blue title
        styling.

    Returns
    -------
    str
        Decorated text, including ANSI colour and reset codes.

    Raises
    ------
    ValueError
        If the text contains 50 or more characters for a title,
        or 30 or more characters for a subtitle.

    # Examples TODO: Unhash
    # -------
    # >>> print(title("Start of Program"))
    # <----------------- Start of Program ----------------->
    #
    # >>> print(title("Section of Program", sub=True))
    # <------ Section of Program ------>
    """
    # Raises a ValueError if the string is too long
    if len(title_string) >= (30 if sub else 50):
        raise ValueError(f"`title_string` must be less than {30 if sub else 50} characters")

    # Make a constant width no matter the length of the title_string
    char_length = int(((30 if sub else 50) - len(title_string)) / 2)
    offset = 1 if len(title_string) % 2 == 1 else 0  # Account for half-spaces

    # Display the title_string in blue with correct length
    colour = "\033[32m" if sub else "\033[94m"
    return colour + "<" + "-" * char_length + f" {title_string} " + "-" * (char_length + offset) + ">\033[0m"


@dataclass()
class ConverterSettings:
    """
    Store image selection and conversion settings.

    Parameters
    ----------
    image_path : pathlib.Path
        Path to an image relative to the working directory.
    image_name : str
        Name of an image file including the extension.
        E.g. ``"test_image.jpg"``.
    colour_depth : int
        Desired number of colours.
    min_contour_area : int
        Minimum area of a group of pixels to retain.
    max_bridge_contour_area : int
        Maximum area of a group of pixels eligible
        for bridging.
    max_contour_distance : int
        Maximum distance between groups of pixels
        that will be bridged.
    bridge_width : int
        Width of a bridge in pixels.
    """
    image_path: Path
    image_name: str
    colour_depth: int
    min_contour_area: int
    max_bridge_contour_area: int
    max_contour_distance: int
    bridge_width: int


def get_inputs(defaults: ConverterSettings,
               image_types: tuple[str, ...],
               image_name: str | None = None,
               colour_depth: int | None = None,
               min_contour_area: int | None = None,
               max_bridge_contour_area: int | None = None,
               max_contour_distance: int | None = None,
               bridge_width: int | None = None,
               ) -> ConverterSettings:
    """
    Collect and validate converter settings, prompting for omitted values.

    Any parameters not passed will be prompted for.

    Parameters
    ----------
    defaults : ConverterSettings
        Default values used when prompting the user for missing values.
    image_types : tuple[str, ...]
        Accepted lowercase file extensions, including the dot.
        E.g. ``(".jpeg", ".jpg", ".png")``.
    image_name : str or None, optional
        Name of an image file including the extension.
        E.g. ``"test_image.jpg"``.
    colour_depth : int or None, optional
        Desired number of colours.
    min_contour_area : int or None, optional
        Minimum area of a group of pixels to retain.
    max_bridge_contour_area : int or None, optional
        Maximum area of a group of pixels eligible
        for bridging.
    max_contour_distance : int or None, optional
        Maximum distance between groups of pixels
        that will be bridged.
    bridge_width : int or None, optional
        Width of a bridge in pixels.

    Returns
    -------
    ConverterSettings
        Validated settings.

    Raises
    ------
    ValueError
        If a passed parameter is invalid.
    TypeError
        If a passed parameter is of the wrong type.
    """

    def validate_image_path_and_name(name: str) -> str:
        """
        Validate a filename based on the following criteria:
        Must not be a path (contain '\\' or '/'),
        end in one of the accepted image types
        and be in the ``images`` folder.
        E.g. ``"test_image.jpg"``.

        Parameters
        ----------
        name : str
            Filename to convert to a string and validate.

        Returns
        -------
        str
            Validated filename, including its extension.

        Raises
        ------
        TypeError
            If string conversion raises ``TypeError``.
        ValueError
            If the name is invalid.
        """
        # Ensure the name is a string
        try:
            name = str(name)
        except TypeError:
            raise TypeError("must be a string")

        # Ensure the name is not a path
        if "\\" in name or "/" in name:
            raise ValueError("cannot be a path (contain '\\' or '/')")

        # Create a Path object from the name using the `images` folder within the working directory
        path = Path("images") / name

        # Ensure the image file is a supported type
        if path.suffix.lower() not in image_types:
            raise ValueError(f"must end in {image_types}")

        # Ensure the image file exists in the `images` folder
        if not path.is_file():
            raise ValueError("was not found in the `images` folder")

        return name

    def receive_image_path_and_name(default: str, value_in: str | None = None) -> tuple[Path, str]:
        """
        Validate a supplied image filename or prompt until one is valid.

        Parameters
        ----------
        default : str
            Filename used in prompts when ``value_in`` is ``None``.
        value_in : str or None
            Filename to validate, ``None`` for interactive input.
            Defaults to ``None``.

        Returns
        -------
        tuple[pathlib.Path, str]
            Validated image path relative to the working directory and filename
            stem without the extension.

        Raises
        ------
        ValueError
            If a passed ``value_in`` fails validation.
        """
        # Loop until a valid image path is received
        while True:
            # If no value is passed, prompt for one
            if value_in is None:
                # Print the title and make sure it's only printed once
                nonlocal printed
                if not printed:
                    print(title("Input Parameters"))
                    printed = True

                # Receive a name from the user. If it is an empty string, use the default name
                value = input(f"Image name [{default}]: ") or default

                # Validate the name, displaying an error message if it fails and allowing the user to try again
                try:
                    value = validate_image_path_and_name(value)
                    break
                except (ValueError, TypeError) as e:
                    # Display the error message using the error message raised by the function
                    print(f"\033[91mError. Image name {e}. Please try again.\033[0m")

            # If a name is passed, use it directly
            else:
                # Validate the name, raising an error if it fails
                try:
                    value = validate_image_path_and_name(value_in)
                    break
                except (ValueError, TypeError) as e:
                    raise ValueError(f"image_name {e}")

        # Return the image path and filename stem without the extension
        return Path("images") / value, str(Path(value).stem)

    def validate_integer(value_in: int | str | bool, min_value_incl: int = 1) -> int:
        """
        Convert a value to an integer and enforce an exclusive lower bound.

        Parameters
        ----------
        value_in : int or str or bool
            Value to convert to an integer and validate.
        min_value_incl : int, optional
            Inclusive minimum allowed value.
            Defaults to 0.

        Returns
        -------
        int
            Validated integer greater than ``min_value_excl``.

        Raises
        ------
        TypeError
            If integer conversion raises ``TypeError``.
        ValueError
            If the integer is invalid.
        OverflowError
            If integer conversion overflows.
        """
        # Ensure the value is an integer
        try:
            value = int(value_in)
        except TypeError:
            raise TypeError("must be an integer")

        # Ensure the value is greater than the minimum allowed value
        if value < min_value_incl:
            raise ValueError(f"must be > {min_value_incl}")

        return value

    def receive_integer(name: str, default: int, value_in: int | None = None, min_value_incl: int = 1,
                        max_warn: int | None = None) -> int:
        """
        Validate a supplied integer or prompt until one is valid.

        Prompted values above ``max_warn`` require confirmation. Supplied
        values are validated without prompting or checking that threshold.

        Parameters
        ----------
        name : str
            Setting name used in prompts, with underscores replaced
            by spaces and capitalised for prompts.
        default : int
            Value used in prompts when ``value_in`` is ``None``.
        value_in : int or None, optional
            Value to validate, ``None`` for interactive input.
        min_value_incl : int, optional
            Inclusive minimum allowed value.
        max_warn : int or None, optional
            Inclusive threshold above which interactive input
            requires confirmation, or ``None`` to disable the warning.

        Returns
        -------
        int
            Validated integer.

        Raises
        ------
        ValueError
            If a passed ``value_in`` is invalid.
        OverflowError
            If integer conversion overflows.
        """
        # Creates a formatted name for the user prompts
        formatted_name = name.replace("_", " ").capitalize()

        # Loop until a valid integer is received
        while True:
            # If no value is passed, prompt for one
            if value_in is None:
                # Print the title and make sure it's only printed once
                nonlocal printed
                if not printed:
                    print(title("Input Parameters"))
                    printed = True

                # Receive a value from the user. If it is an empty string, use the default value
                value = input(f"{formatted_name} [{default}]: ") or default

                # Validate the value, displaying an error message if it fails and allowing the user to try again
                try:
                    value = validate_integer(value, min_value_incl)

                    # If there is a maximum recommended value, and it had been exceeded, then prompt for confirmation, warning the user
                    if max_warn is not None and value > max_warn:
                        # Loop until 'Y' or 'N' is entered
                        while True:
                            answer = input(
                                f"\033[33mWarning. {formatted_name} is not recommended to be over{max_warn}. Are you sure you want to proceed (Y/[N])? \033[0m") or "N"
                            # Ensure the user entered 'Y' or 'N'
                            try:
                                # If the user entered 'N', let them re-enter the value
                                if not validate_boolean(answer):
                                    continue
                            except ValueError as e:
                                print(f"\033[91mError. Invalid input, {e}. Please try again.\033[0m")
                                continue
                            break
                    return value

                # Display the error message using the error message raised by the function
                except (ValueError, TypeError) as e:
                    print(f"\033[91mError. {formatted_name} {e}. Please try again.\033[0m")

            # If a value is passed, use it directly
            else:
                # Validate the name, raising an error if it fails
                try:
                    return validate_integer(value_in, min_value_incl)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"{name} {e}")

    def validate_boolean(value_in: str | bool) -> bool:
        """
        Ensure a passed value is either ``'Y'``, ``'N'`` or ``bool``.

        Parameter
        ---------
        value_in : str | bool
            Value to be validated.
        Returns
        -------
        bool
            Validated boolean.

        Raises
        ------
        ValueError
            If the value is not ``'Y'`` or ``'N'``.
        TypeError
            If the passed value is not a string or bool.
        """
        if type(value_in) is bool:
            return bool(value_in)
        # Ensure the value is 'Y' or 'N'
        if str(value_in).upper() not in ("Y", "N"):
            raise ValueError(f"must be 'Y' or 'N' not {value_in}")

        return False if str(value_in).upper() == "N" else True

    def receive_boolean(name: str, default: str | bool, value_in: str | bool | None = None) -> bool:
        """
        Validate a supplied boolean or prompt until one is valid.

        Parameters
        ----------
        name: str
            Setting name used in prompts.
        default: str | bool
            Value used in prompts when ``value_in`` is ``None``.
        value_in: str | bool | None, optional
            Value to validate, ``None`` for interactive input.
            Defaults to ``None``.

        Returns
        -------
        bool
            Validated boolean.

        Raises
        ------
        ValueError
            If a passed ``value_in`` is invalid.

        """
        # Creates a formatted name and default for the user prompts
        formatted_name = name.replace("_", " ").capitalize()
        formatted_default = f"(Y/[N])" if default == ("N" if type(default) is str else False) else f"(Y/[N])"

        while True:
            if value_in is None:
                nonlocal printed
                if not printed:
                    print(title("Input Parameters"))
                    printed = True

                value = input(f"{formatted_name} {formatted_default}? ") or validate_boolean(default)

                try:
                    return validate_boolean(value)
                except ValueError as e:
                    print(f"\033[91mError. Invalid input, {e}. Please try again.\033[0m")

            else:
                try:
                    return validate_boolean(value_in)
                except ValueError as e:
                    raise ValueError(f"{name} {e}")

    # Initialise the `printed` variable to False
    printed = False

    # Receive all the settings using the appropriate defaults, mins and maxes
    image_path, image_name = receive_image_path_and_name(defaults.image_name, image_name)
    colour_depth = receive_integer("colour_depth", defaults.colour_depth, colour_depth, min_value_incl=2, max_warn=50)
    min_contour_area = receive_integer("min_contour_area", defaults.min_contour_area, min_contour_area)
    max_bridge_contour_area = receive_integer("max_bridge_contour_area", defaults.max_bridge_contour_area,
                                              max_bridge_contour_area, min_value_incl=0)
    max_contour_distance = receive_integer("max_contour_distance", defaults.max_contour_distance,
                                           max_contour_distance, min_value_incl=0)
    bridge_width = receive_integer("bridge_width", defaults.bridge_width, bridge_width, max_warn=25)

    # Return the validated settings as a dataclass
    return ConverterSettings(
        image_path=image_path,
        image_name=image_name,
        colour_depth=colour_depth,
        min_contour_area=min_contour_area,
        max_bridge_contour_area=max_bridge_contour_area,
        max_contour_distance=max_contour_distance,
        bridge_width=bridge_width,
    )


class Converter:
    DEFAULTS = ConverterSettings(
        image_path=Path("images") / "test_image.jpg",
        image_name="test_image.jpg",
        colour_depth=8,
        min_contour_area=30,
        max_bridge_contour_area=50,
        max_contour_distance=100,
        bridge_width=1,
    )

    IMAGE_TYPES = (".jpeg", ".jpg", ".png")

    def __init__(self, **kwargs: dict[str, int | None]):
        """
        Initialise the converter, prompting for settings that are omitted.

        Parameters
        ----------
        **kwargs
            Optional arguments passed to :func:`get_inputs`:
            See :func:`get_inputs` for details.

        Raises
        ------
        ValueError
            If a keyword argument is invalid.
        TypeError
            If a keyword argument is of the wrong type.
        """
        # Initialise settings using the passed arguments or prompting the user if one isn't passed
        self.settings: ConverterSettings = get_inputs(self.DEFAULTS, self.IMAGE_TYPES, **kwargs)

        # Initialise the images as none and specify their type as being a ndarray of type uint8
        self.original_image: npt.NDArray[np.uint8] | None = None
        self.image_no_bg: npt.NDArray[np.uint8] | None = None
        self.quantised_image: npt.NDArray[np.uint8] | None = None
        self.pixel_labels: npt.NDArray[np.int32] | None = None

        self.output_path: Path = Path("")

        self.binary_layers: list[npt.NDArray[np.uint8] | None] = []

    def load_image(self):
        """
        Load an image as an RGBA array, applying its EXIF orientation.
        Saves the image as a class attribute of type ``numpy.ndarray`` of shape ``(height, width, 4)``.

        Raises
        ------
        FileNotFoundError
            If the image path does not exist.
        PIL.UnidentifiedImageError
            If Pillow cannot identify the image.
        OSError
            If the image cannot be opened or decoded.
        """
        print("Loading image...", end="")
        img = Image.open(self.settings.image_path)  # Open the image
        img = ImageOps.exif_transpose(img)  # Ensure image is correctly orientated
        img = img.convert("RGBA")  # Convert the image to RGBA
        img_array = np.array(img, dtype=np.uint8)  # Convert the image to a numpy array of pixels
        print("\rLoaded Image.")

        # Save the image as a class attribute
        self.original_image = img_array

    def create_output_folder(self):
        """
        Create an empty output folder based on the image name.
        Any existing file or directory at the output path is deleted,
        including all contents of an existing directory.
        Saves the output path as a class attribute of type ``pathlib.Path``.

        Raises
        ------
        OSError
            If an existing output cannot be removed or the new
            directory cannot be created.
        """
        deleted = False
        print("Creating output folder...", end="")
        # Create a folder path based on the image name
        output_path = Path(f"{self.settings.image_name}_output")

        # If the folder already exists, delete it
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
            deleted = True

        # Create the new folder
        output_path.mkdir(parents=True, exist_ok=False)
        print(f"\rCreated output folder{" and deleted the existing folder." if deleted else "."}")

        # Save the output path as a class attribute
        self.output_path = output_path

    def remove_background(self):
        """
        Replaces any background pixels with black.
        A background pixel is any pixel with any transparency (alpha < 255).
        Saves the image as a class attribute of type ``numpy.ndarray`` of shape ``(height, width, 3)``.

        Raises
        ------
        ValueError
            If no image has been loaded into the ``original_image`` attribute.
        """
        if self.original_image is None:
            raise ValueError("image not loaded, please load an image before trying to quantise it")

        print("Selecting transparent pixels...", end="")
        alpha_channel = self.original_image[..., 3].copy()  # Select the alpha channel
        # Create a boolean array of the transparent pixels
        trans_pixels = alpha_channel < 255
        # Count the number of transparent pixels
        n_trans_pixels = int(np.sum(trans_pixels))
        print(f"\rSelected {n_trans_pixels} transparent pixels.")

        print("Replacing transparent pixels with black...", end="")
        # Select the RGB channels of the image
        image = self.original_image[..., :3].copy()
        # Replace the transparent pixels with black
        image[trans_pixels] = [0, 0, 0]
        print(f"\rReplaced {n_trans_pixels} transparent pixels with black.")

        print("Saving quantised image...", end="")
        # Save the quantised image and pixel labels
        cv2.imwrite(str(self.output_path / f"removed_bg_{self.settings.image_name}.jpg"), image)
        self.image_no_bg = image
        print("\rSaved quantised image.")

    def quantise_image(self):
        """
        Removes background pixels and quantises the image.
        Removes any pixels with any amount of transparency.
        Creates a new image that only contains the main colours of the original image (quantisation).
        Saves the quantised image as a ``.jpg`` file in the BRG colour space.
        Saves the quantised image as a class attribute of type ``numpy.ndarray`` of shape ``(height, width, 3)``.

        Raises
        ------
        ValueError
            If no image has been loaded into the ``original_image`` attribute.
        """
        if self.original_image is None:
            raise ValueError("image not loaded, please load an image before trying to quantise it")

        print(f"Selecting {self.settings.colour_depth} colours...", end="")
        image_pil_rgb = Image.fromarray(self.image_no_bg, mode="RGB")
        image_pil_quantised_rgb = image_pil_rgb.quantize(colors=self.settings.colour_depth,
                                                         method=Image.Quantize.FASTOCTREE,
                                                         dither=Image.Dither.NONE
                                                         )
        height, width = self.image_no_bg.shape[:2]

        pixel_labels = np.asarray(image_pil_quantised_rgb, dtype=np.int32).reshape(height, width)

        palette_data = image_pil_quantised_rgb.getpalette()
        palette_rgb = np.asarray(palette_data, dtype=np.uint8).reshape(-1, 3)
        colour_groups_hls = cv2.cvtColor(palette_rgb.reshape(1, -1, 3), cv2.COLOR_RGB2HLS).reshape(-1, 3).astype(np.float32)

        # Old K-Means algorithm
        # print("Converting image to HLS colour space...", end="")
        # # Convert the image to HLS colour space
        # image_hls = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS)
        # print("\rConverted image to HLS colour space.")
        #
        # print("Reshaping image...", end="")
        # # Get the height and width of the image in pixels
        # height, width = image_hls.shape[:2]
        # # Flatten the array so each pixel has an index with 3 colours
        # image_flat_hls = image_hls.reshape((-1, 3)).astype(np.float32)
        # print(f"\rReshaped image.")
        #
        # print(f"Selecting {self.settings.colour_depth} colours...", end="")
        # # Define the criteria for stopping the K-Means clustering algorithm
        # stop_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
        #
        # # Use a sample of the image if best colour quality is not selected, otherwise use all pixels
        # if self.settings.best_colour_quality:
        #     sample = image_flat_hls
        # else:
        #     sample_size = min(10000, len(image_flat_hls))
        #     indices = np.random.choice(len(image_flat_hls), sample_size, replace=False)
        #     sample = image_flat_hls[indices]
        #
        # # Use K-Means clustering to select the desired number of colours
        # _, labels, centers = cv2.kmeans(sample, self.settings.colour_depth, np.empty((0, 1)), stop_criteria, 1, cv2.KMEANS_PP_CENTERS)
        #
        # # Store the cluster centres (chosen colours)
        # colour_groups = centers.astype(np.float32)
        #
        # # Assigns each pixel to a colour group based on the closest colour in the cluster
        # if self.settings.best_colour_quality:
        #     pixel_labels = labels.ravel().astype(np.int32)
        # else:
        #     distances = np.sum((image_flat_hls[:, np.newaxis, :] - colour_groups[np.newaxis, :, :]) ** 2, axis=2)
        #     pixel_labels = np.argmin(distances, axis=1).astype(np.int32)
        # print(f"\rSelected {self.settings.colour_depth} colours.")

        print("Sorting colours by lightness...", end="")
        # Sort the colours by their lightness so that the lighter colours are first
        colour_groups_order = (np.argsort(colour_groups_hls[:, 1]))
        ordered_colour_groups = colour_groups_hls[colour_groups_order]

        # Reassign the pixel labels to the new order
        blank_image = np.empty_like(colour_groups_order)
        blank_image[colour_groups_order] = np.arange(len(colour_groups_order))
        ordered_pixel_labels = blank_image[pixel_labels].astype(np.int32)
        print("\rSorted colours by lightness.")

        print("Rebuilding image...", end="")
        # Rebuild the image from the pixel labels and colour groups
        quantised_image = ordered_colour_groups[ordered_pixel_labels].astype(np.uint8)
        quantised_image = quantised_image.reshape((height, width, 3))
        print("\rRebuilt image.")

        print("Converting image to BGR colour space...", end="")
        # noinspection bad-assignment
        quantised_image_bgr: npt.NDArray[np.uint8] = cv2.cvtColor(quantised_image, cv2.COLOR_HLS2BGR)
        print("\rConverted image to BGR colour space.")

        print("Saving quantised image...", end="")
        # Save the quantised image and pixel labels
        cv2.imwrite(str(self.output_path / f"quantised_{self.settings.image_name}.jpg"), quantised_image_bgr)
        self.quantised_image = quantised_image_bgr
        self.pixel_labels = ordered_pixel_labels.reshape(height, width)
        print("\rSaved quantised image.")

    def build_binary_layers(self):
        print("Creating output folder for binary layers...", end="")
        output_path = self.output_path / "_binary_layers"
        output_path.mkdir(parents=True, exist_ok=False)
        print("\rCreated output folder for binary layers.")

        layer = np.zeros_like(self.quantised_image)

        for i in range(self.settings.colour_depth):
            print(f"Converting each colour to a binary layer... [{i + 1}/{self.settings.colour_depth}]", end=" ")

            if i > 0:
                layer[self.pixel_labels == i - 1] = 255


converter = Converter()

start = time.perf_counter()
print(title("Load Image & Create Output Folder"))
converter.load_image()
converter.create_output_folder()

print(title("Remove Background"))
converter.remove_background()

print(title("Quantise Image"))
converter.quantise_image()

end_time = time.perf_counter() - start
print(f"\nFinished in {end_time:.2f} seconds.")

# ---- Image2Vector ----
#  ---- 02/08/2026 ----
import shutil
import time
from dataclasses import dataclass, astuple
from pathlib import Path

import cv2
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
               all_settings: ConverterSettings | None = None,
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
    all_settings : ConverterSettings or None, optional
        A dataclass containing all settings.
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

    if all_settings is not None:
        _, image_name, colour_depth, min_contour_area, max_bridge_contour_area, max_contour_distance, bridge_width = astuple(all_settings)

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


class Images:
    """
    Store the various images used in the program.
    See :class:`Converter` for the methods used to manipulate these images.

    Parameters
    ----------
    original_image: npt.NDArray[np.uint8]
        The original image that should be loaded from the ``images`` folder as specified in the settings.
    image_no_bg: npt.NDArray[np.uint8]
        The image with its background pixels replaced with black.
    quantised_image: npt.NDArray[np.uint8]
        The image after it had been quantised using the ``colour_depth`` settings.
    pixel_labels: npt.NDArray[np.int32]
    """
    # Images are stored as numpy arrays of type uint8
    original_image: npt.NDArray[np.uint8] = np.empty((1, 1, 4)).astype(np.uint8)  # RBGA
    image_no_bg: npt.NDArray[np.uint8] = np.empty((1, 1, 3)).astype(np.uint8)  # RGB
    quantised_image: npt.NDArray[np.uint8] = np.empty((1, 1, 3)).astype(np.uint8)  # BGR
    pixel_labels: npt.NDArray[np.int32] = np.empty((1, 1)).astype(np.int32)

    # A list of the binary layers of the quantised image, one for each colour
    binary_layers: list[npt.NDArray[np.uint8]] = [np.empty((1, 1, 3)).astype(np.uint8)]


class Converter:
    DEFAULTS = ConverterSettings(
        image_path=Path("images") / "camping_rusty_field.png",
        image_name="camping_rusty_field.png",
        colour_depth=8,
        min_contour_area=30,
        max_bridge_contour_area=50,
        max_contour_distance=100,
        bridge_width=1,
    )

    IMAGE_TYPES = (".jpeg", ".jpg", ".png")

    def __init__(self, **kwargs):
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

        self.output_path: Path = Path("")

    def load_image(self) -> npt.NDArray[np.uint8]:
        """
        Load an image as an RGBA array, applying its EXIF orientation.

        Returns
        -------
        npt.NDArray[np.uint8]
            The loaded image as an RGBA array.

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

        return img_array

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

    def remove_background(self, og_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Replaces any background pixels with black.
        A background pixel is any pixel with any transparency (alpha < 255).

        Returns
        -------
        npt.NDArray[np.uint8]
            The image with its background pixels replaced with black as an RGB array.
        """

        print("Selecting transparent pixels...", end="")
        alpha_channel = og_image[..., 3].copy()  # Select the alpha channel
        # Create a boolean array of the transparent pixels
        trans_pixels = alpha_channel < 255
        # Count the number of transparent pixels
        n_trans_pixels = int(np.sum(trans_pixels))
        print(f"\rSelected {n_trans_pixels} transparent pixels.")

        print("Replacing transparent pixels with black...", end="")
        # Select the RGB channels of the image
        image = og_image[..., :3].copy()
        # Replace the transparent pixels with black
        image[trans_pixels] = [0, 0, 0]
        print(f"\rReplaced {n_trans_pixels} transparent pixels with black.")

        print("Saving image...", end="")
        # Save image and pixel labels
        cv2.imwrite(str(self.output_path / f"removed_bg_{self.settings.image_name}.jpg"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("\rSaved image.")

        return image

    def quantise_image(self,
                       og_image: npt.NDArray[np.uint8],
                       hue_bins: int = 16,
                       lightness_bins: int = 3,
                       saturation_bins: int = 3,
                       log_base: float | int = 1.5,
                       saturation_strength: float | int = 0.7
                       ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int32]]:
        """
        Removes background pixels and quantises the image using octree quantisation.
        Removes any pixels with any amount of transparency.
        Creates a new image that only contains the main colours of the original image (quantisation).

        Returns
        -------
        npt.NDArray[np.uint8]
            The quantised image as an BGR array.
        npt.NDArray[np.int32]
            An array of labelled pixels.

        Raises
        ------
        ValueError
            if ``log_base`` is 1 or less.
        """
        if log_base <= 1:
            raise ValueError("log_base must be greater than 1")

        print("Creating a copy of the image in HLS colour space...", end="")
        image_hls = cv2.cvtColor(og_image, cv2.COLOR_RGB2HLS)
        print("\rCreated a copy of the image in HLS colour space.")

        print("Binning pixels into colour groups...", end="")
        # Select the HLS channels of the image, each array gives each pixel one value for the channel
        image_hue = image_hls[..., 0].astype(np.int32)
        image_light = image_hls[..., 1].astype(np.int32)
        image_sat = image_hls[..., 2].astype(np.int32)

        # Place each pixel into the correct channel bin
        image_hue_binned = np.clip(image_hue * hue_bins // 180, 0, hue_bins - 1)
        image_lightness_binned = np.clip(image_light * lightness_bins // 256, 0, lightness_bins - 1)
        image_saturation_binned = np.clip(image_sat * saturation_bins // 256, 0, saturation_bins - 1)

        # Combine the bins where each combination of HLS bins has a unique id
        image_binned_id_flat = (image_hue_binned * lightness_bins * saturation_bins
                                + image_lightness_binned * saturation_bins
                                + image_saturation_binned
                                ).reshape(-1)
        print("\rBinned pixels into colour groups.")

        print("Getting the mean RGB values of each bin...", end="")
        # Get the present bins, inverse indices (Index of present bin in image_binned_id_flat) and counts of each bin
        present_bins, inverse_indices, counts = np.unique(image_binned_id_flat, return_inverse=True, return_counts=True)

        image_rgb_flat = og_image.reshape(-1, 3).astype(np.uint8)

        # Create an array to store the sum of the colours of the pixels in each bin
        empty_bin_colours = np.zeros((len(present_bins), 3), dtype=np.float32)

        # Sum the colours of the pixels in each bin
        np.add.at(empty_bin_colours, inverse_indices, image_rgb_flat)

        # Divide each bin's sum by the number of pixels in that bin
        mean_bin_colours = empty_bin_colours / counts[:, None]

        # Ensure the mean colours are within the range 0-255
        mean_bin_colours = np.rint(np.clip(mean_bin_colours, 0, 255)).astype(np.uint8)

        # Increase the saturation to improve visual similarity
        mean_bin_colours_hls = cv2.cvtColor(mean_bin_colours[None, :, :], cv2.COLOR_RGB2HLS)[0].astype(np.float32)

        saturation = mean_bin_colours_hls[..., 2] / 255.0
        # Tail off the saturation near the extremes
        boosted_saturation = saturation + (saturation_strength * saturation * (1.0 - saturation))

        mean_bin_colours_hls[..., 2] = np.clip(boosted_saturation * 255.0, 0, 255).astype(np.float32)
        mean_bin_colours = cv2.cvtColor(np.rint(mean_bin_colours_hls).astype(np.uint8)[None, :, :], cv2.COLOR_HLS2RGB)[
            0]
        print("\rGot the mean RGB values of each bin.")

        print("Creating a weighted array of the colour bins...", end="")
        # Weight counts to reduce the bias in frequent colours
        weighted_counts = 1.0 + (np.log(counts) / np.log(log_base))
        weighted_counts = np.maximum(1, np.rint(weighted_counts)).astype(np.int32)

        # Create an array where the colour of each bin is shown its weighted_counts times
        weighted_image = np.repeat(mean_bin_colours, weighted_counts, axis=0)
        print("\rCreated a weighted array of the colour bins.")

        print(f"Selecting {self.settings.colour_depth} colours...", end="")
        # Turn the array into an image for Pillow to quantise
        weighted_pil_image_rgb = Image.fromarray(weighted_image[None, :, :], mode="RGB")
        # Quantise the weighted counts image using the specified number of colours
        weighted_pil_image_quantised_rgb = weighted_pil_image_rgb.quantize(colors=self.settings.colour_depth,
                                                                           method=Image.Quantize.FASTOCTREE,
                                                                           dither=Image.Dither.NONE
                                                                           )

        # pixel_labels = np.asarray(image_pil_quantised_rgb, dtype=np.int32).reshape(height, width)

        # Get the colours of the quantised image
        palette_data = weighted_pil_image_quantised_rgb.getpalette()

        # Ensure palette_rgb is only that of colours used in the weighted_pil_image_quantised_rgb
        weighted_pixel_labels = np.asarray(weighted_pil_image_quantised_rgb, dtype=np.int32).reshape(-1)
        palette_rgb_all = np.asarray(palette_data, dtype=np.uint8).reshape(-1, 3)
        used_palette_indices = np.unique(weighted_pixel_labels)
        palette_rgb = palette_rgb_all[used_palette_indices].astype(np.uint8).reshape(-1, 3)

        # Assign each pixel to the nearest colour in the palette
        distances = np.sum((image_rgb_flat.astype(np.float32)[:, None, :]
                            - palette_rgb.astype(np.float32)[None, :, :]
                            ) ** 2, axis=2)
        pixel_labels = np.argmin(distances, axis=1).astype(np.int32)

        # Convert the palette into colour groups to be sorted in HLS
        colour_groups_hls = cv2.cvtColor(palette_rgb[None, :, :], cv2.COLOR_RGB2HLS)[0].astype(np.float32)

        print("Sorting colours by lightness...", end="")
        # Sort the colours by their lightness so that the lighter colours are first
        colour_groups_order = (np.argsort(colour_groups_hls[:, 1]))

        # Reassign the pixel labels to the new order
        ordered_palette_rgb = palette_rgb[colour_groups_order]

        palette_index_map = np.empty_like(colour_groups_order)
        palette_index_map[colour_groups_order] = np.arange(len(colour_groups_order), dtype=np.int32)

        ordered_pixel_labels = palette_index_map[pixel_labels].astype(np.int32)
        print("\rSorted colours by lightness.")

        print("Rebuilding image...", end="")
        # Rebuild the image from the pixel labels and colour groups
        height, width = og_image.shape[:2]
        quantised_image_rgb = ordered_palette_rgb[ordered_pixel_labels].reshape(height, width, 3)
        print("\rRebuilt image.")

        print("Converting image to BGR colour space...", end="")
        # noinspection bad-assignment
        quantised_image_bgr: npt.NDArray[np.uint8] = cv2.cvtColor(quantised_image_rgb, cv2.COLOR_RGB2BGR)
        print("\rConverted image to BGR colour space.")

        print("Saving quantised image...", end="")
        # Save the quantised image and pixel labels
        cv2.imwrite(str(self.output_path / f"quantised_{self.settings.image_name}.jpg"), quantised_image_bgr)
        print("\rSaved quantised image.")

        return quantised_image_bgr, ordered_pixel_labels.reshape(height, width)

    # def build_binary_layers(self):
    #     print("Creating output folder for binary layers...", end="")
    #     output_path = self.output_path / "_binary_layers"
    #     output_path.mkdir(parents=True, exist_ok=False)
    #     print("\rCreated output folder for binary layers.")
    #
    #     layer = np.zeros_like(self.quantised_image)
    #
    #     for i in range(self.settings.colour_depth):
    #         print(f"Converting each colour to a binary layer... [{i + 1}/{self.settings.colour_depth}]", end=" ")
    #
    #         if i > 0:
    #             layer[self.pixel_labels == i - 1] = 255


converter = Converter(all_settings=ConverterSettings(
    image_path=Path("images") / "camping_rusty_field.png",
    image_name="camping_rusty_field.png",
    colour_depth=8,
    min_contour_area=30,
    max_bridge_contour_area=50,
    max_contour_distance=100,
    bridge_width=1,
))

images = Images()

start = time.perf_counter()
print(title("Load Image & Create Output Folder"))
images.original_image = converter.load_image()
converter.create_output_folder()

print(title("Remove Background"))
images.image_no_bg = converter.remove_background(images.original_image)

print(title("Quantise Image"))
images.quantised_image, images.pixel_labels = converter.quantise_image(images.image_no_bg)

end_time = time.perf_counter() - start
print(f"\nFinished in {end_time:.2f} seconds.")

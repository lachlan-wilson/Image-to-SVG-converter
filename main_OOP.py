# ---- Image2Vector ----
#  ---- 02/08/2026 ----
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps
from sklearn.cluster import MiniBatchKMeans

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

    def validate_integers(value_in: int | str | bool, min_value_incl: int = 1) -> int:
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
        value_in : int or None
            Value to validate, ``None`` for
            interactive input.
            Defaults to ``None``.
        min_value_incl : int, optional
            Inclusive minimum allowed value.
            Defaults to 0.
        max_warn : int or None, optional
            Inclusive threshold above which interactive input
            requires confirmation, or ``None`` to disable the warning.
            Defaults to ``None``.

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
                    value = validate_integers(value, min_value_incl)

                    # If there is a maximum recommended value, and it had been exceeded, then prompt for confirmation, warning the user
                    if max_warn is not None and value > max_warn:
                        # Loop until 'Y' or 'N' is entered
                        while True:
                            answer = input(
                                f"\033[33mWarning. {formatted_name} is not recommended to be over{max_warn}. Are you sure you want to proceed (Y/[N])? \033[0m") or "N"
                            if answer.upper() not in ("Y", "N"):
                                print(f"\033[91mError. Invalid input. Please enter 'Y' or 'N'.\033[0m")
                                continue
                            break
                        # If the user entered 'N', let them re-enter the value
                        if answer.upper() == "N":
                            continue
                    return value

                # Display the error message using the error message raised by the function
                except (ValueError, TypeError) as e:
                    print(f"\033[91mError. {formatted_name} {e}. Please try again.\033[0m")

            # If a value is passed, use it directly
            else:
                # Validate the name, raising an error if it fails
                try:
                    return validate_integers(value_in, min_value_incl)
                except (ValueError, TypeError) as e:
                    if not e[1]:
                        raise ValueError(f"{name} {e[0]}")
        return 0

    # Initialise the `printed` variable to False
    printed = False

    # Receive all the settings using the appropriate defaults, mins and maxes
    image_path, image_name = receive_image_path_and_name(defaults.image_name, image_name)
    colour_depth = receive_integer("colour_depth", defaults.colour_depth, colour_depth, max_warn=50)
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
        self.quantised_image: npt.NDArray[np.uint8] | None = None

        self.output_path: Path = Path("")

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
        img_array = np.array(img)  # Convert the image to a numpy array of pixels
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
        output_path = Path(self.settings.image_name + "_output")

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

    def quantise_image(self):
        print("Selecting transparent pixels...", end="")
        if self.original_image is None:
            raise ValueError("image not loaded, please load an image before trying to quantise it")

        alpha_channel = self.original_image[..., 3]  # Select the alpha channel
        # Create a boolean array of the transparent pixels
        trans_pixels = np.ravel(alpha_channel < 255)
        print(f"\rSelected {len(trans_pixels)} transparent pixels.")

        print("Converting the image to HLS colour space...", end="")
        # Convert the image to HLS colour space
        image_hls = cv2.cvtColor(self.original_image[..., :3], cv2.COLOR_RGB2HLS)
        print("\rConverted the image to HLS colour space.")

        print("Reshaping the image...", end="")
        # Get the height and width of the image in pixels
        height, width = image_hls.shape[:2]
        n_pixels = height * width
        # Flatten the array so each pixel has an index with 3 colours,
        image_flat_hls = image_hls.reshape((-1, 3)).astype(np.uint8)
        print(f"\rReshaped the image to {n_pixels} pixels.")

        print("Removing transparent pixels...", end="")
        n_trans_pixels = len(trans_pixels)
        trans_exist = n_trans_pixels > 0
        image_flat_hls_no_bg = image_flat_hls[~trans_pixels]
        print(f"\rRemoved {n_trans_pixels} transparent pixels.")

        print(f"Selecting {self.settings.colour_depth} colours...", end="")
        # TODO: Change kmeans to be better
        kmeans = MiniBatchKMeans(n_clusters=(self.settings.colour_depth - int(trans_exist)), random_state=42, batch_size=2048)
        image_pixel_labels = kmeans.fit_predict(image_flat_hls_no_bg)
        colour_groups = kmeans.cluster_centers_
        print(f"\rSelected {self.settings.colour_depth} colours.")

        if trans_exist:
            print("Adding a background...", end="")
            pixel_labels = np.full(n_pixels, fill_value=(self.settings.colour_depth - 1), dtype=np.int8)
            pixel_labels[~trans_pixels] = pixel_labels
            black = np.array([0, 0, 0], dtype=np.uint8)
            colour_groups = np.vstack((colour_groups, black))
            print("\rAdded a background.")

        print("Sorting colours by lightness...", end="")
        colour_groups_order = (np.argsort(colour_groups[:, 1]))
        ordered_colour_groups = colour_groups[colour_groups_order]

        blank_image = np.zeros_like(colour_groups_order)
        blank_image[colour_groups_order] = np.arange(len(colour_groups_order))
        ordered_pixel_labels = blank_image[image_pixel_labels]
        print("\rSorted colours by lightness.")

        print("Rebuilding the image...", end="")
        quantised_image = ordered_colour_groups[ordered_pixel_labels].astype(np.uint8)
        quantised_image = quantised_image.reshape((height, width, 3))
        print("\rRebuilt the image.")

        print("Converting the image to BGR colour space...", end="")
        quantised_image_bgr = cv2.cvtColor(quantised_image, cv2.COLOR_HLS2BGR)
        print("\rConverted the image back to BGR colour space.")

        print("Saving the quantised image...", end="")
        cv2.imwrite(self.output_path / f"quantised_{self.settings.image_name}.jpg", quantised_image_bgr)
        self.quantised_image = quantised_image_bgr
        print("\rSaved the quantised image.")


converter = Converter()

print(title("Load Image & Create Output Folder"))
converter.load_image()
converter.create_output_folder()

print(title("Quantise Image"))
converter.quantise_image()

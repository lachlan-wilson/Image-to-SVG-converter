# ---- Image2Vector ----
#  ---- 02/08/2026 ----
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy
from PIL import Image, ImageOps

POTRACE_PATH = "/usr/local/bin/potrace"  # Absolute path of Potrace


def title(title_string: str, sub: bool = False) -> str:
    """
    Turn a string into a title with a constant width.

    Parameters
    ----------
    title_string: str
        The string to be formatted.
    sub: bool, default=False
        Whether the string is a subtitle.

    Returns
    -------
    str
        The formatted string.

    Example
    ------
    """
    # Raises a ValueError if the string is too long
    if len(title_string) >= (30 if sub else 50):
        raise ValueError(f"`title_string` must be less than {30 if sub else 50} characters")

    # Make a constant width no matter the length of the title_string
    char_length = int((30 if sub else 50 - len(title_string)) / 2)
    offset = 1 if len(title_string) % 2 == 1 else 0  # Account for half-spaces

    # Display the title_string in blue with correct length
    colour = "\033[32m" if sub else "\033[94m"
    return colour + "<" + "-" * char_length + f" {title_string} " + "-" * (char_length + offset) + ">\033[0m"


@dataclass()
class ConverterSettings:
    """
    A dataclass containing the settings for the converter.
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
    Get all the validated input parameters from the user.

    Returns
    -------
    None
    """

    def validate_image_path_and_name(name: str) -> str:
        try:
            name = str(name)
        except (ValueError, TypeError):
            raise TypeError("must be a string")

        if "\\" in name or "/" in name:
            raise ValueError("cannot be a path (contain '\\' or '/')")

        path = Path("images") / name

        if path.suffix.lower() not in image_types:
            raise ValueError(f"must end in {image_types}")

        if not path.is_file():
            raise ValueError("was not found in the `images` folder")

        return name

    def receive_image_path_and_name(default: str, value_in: str | None = None) -> tuple[Path, str]:
        while True:
            if value_in is None:
                nonlocal printed
                if not printed:
                    print(title("Input Parameters"))
                    printed = True
                value = input(f"Image name [{default}]: ") or default

                try:
                    value = validate_image_path_and_name(value)
                    break
                except (ValueError, TypeError) as e:
                    print(f"\033[91mError. Image name {e}. Please try again.\033[0m")

            else:
                try:
                    value = validate_image_path_and_name(value_in)
                    break
                except (ValueError, TypeError) as e:
                    raise ValueError(f"image_name {e}")

        return Path("images") / value, str(Path(value).stem)

    def validate_integers(value_in, min_value_excl: int = 0) -> int:
        try:
            value = int(value_in)
        except (ValueError, TypeError):
            raise TypeError("must be an integer")

        if min_value_excl >= value:
            raise ValueError(f"must be > {min_value_excl}")

        return value

    def receive_integer(name: str, default: int, value_in: int | None = None, min_value_excl=0, max_warn=None) -> int:
        formatted_name = name.replace("_", " ").capitalize()
        while True:
            if value_in is None:
                nonlocal printed
                if not printed:
                    print(title("Input Parameters"))
                    printed = True
                value = input(f"{formatted_name} [{default}]: ") or default

                try:
                    value = validate_integers(value, min_value_excl)
                    if max_warn is not None and value > max_warn:
                        while True:
                            answer = input(
                                f"\033[33mWarning. {formatted_name} is not recommended to be over{max_warn}. Are you sure you want to proceed (Y/[N])? \033[0m") or "N"
                            if answer.upper() not in ("Y", "N"):
                                print(f"\033[91mError. Invalid input. Please enter 'Y' or 'N'.\033[0m")
                                continue
                            break
                        if answer.upper() == "N":
                            continue
                    return value

                except (ValueError, TypeError) as e:
                    print(f"\033[91mError. {formatted_name} {e}. Please try again.\033[0m")
            else:
                try:
                    return validate_integers(value_in, min_value_excl)
                except (ValueError, TypeError) as e:
                    if not e[1]:
                        raise ValueError(f"{name} {e[0]}")
        return 0

    printed = False

    image_path, image_name = receive_image_path_and_name(defaults.image_name, image_name)
    colour_depth = receive_integer("colour_depth", defaults.colour_depth, colour_depth, max_warn=50)
    min_contour_area = receive_integer("min_contour_area", defaults.min_contour_area, min_contour_area)
    max_bridge_contour_area = receive_integer("max_bridge_contour_area", defaults.max_bridge_contour_area,
                                              max_bridge_contour_area)
    max_contour_distance = receive_integer("max_contour_distance", defaults.max_contour_distance, max_contour_distance)
    bridge_width = receive_integer("bridge_width", defaults.bridge_width, bridge_width, max_warn=25)

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

    def __init__(self, **kwargs):
        # Initialise variables
        self.settings = get_inputs(self.DEFAULTS, self.IMAGE_TYPES, **kwargs)

    def load_image(self, img_path):
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
        print("Loading image...", end="")
        img = Image.open(img_path)  # Open the image
        img = ImageOps.exif_transpose(img)  # Ensure image is correctly orientated
        img = img.convert("RGBA")  # Convert the image to RGBA
        img = numpy.array(img)  # Convert the image to a numpy array of pixels
        print("\rLoaded Image.\n")

        return img

    def create_output_folder(self, img_name):
        """
        Create an output folder with the same name as the image.
        Overwrites any existing folder with that name.

        Parameters
        ----------
        img_name: str
            The name of the image.

        Returns
        -------
        `Path`
            The path to the output folder.
        """
        print("Creating output folder...", end="")
        output_path = Path(img_name + "_output")

        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()

        output_path.mkdir(parents=True, exist_ok=False)
        print("\rCreated output folder.\n")

        return output_path


converter = Converter()

print(converter.settings)

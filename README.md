# Image to SVG converter
## Overview
This program was designed to easily turn images into SVGs that could then be used to make paper cut posters. It takes an image and converts it into an SVG. It first turns the image into layers of the most common colours, then it cleans these layers, removing small contours (groups of pixels) and bridging slightly larger ones. Using Potrace it then turns each layer into an SVG. The layers are then combined into one SVG.
## Example
![Before and After](./examples/boat_before_and_after.png)
[Example Image](./examples/boat.jpg)    [Example SVG](./examples/boat_combined.svg)
## Usage
### Inputting an Image
- The image must be within the `Images` folder
- The image must be one of the following: `.jpeg`, `.jpg`, `.png`
- When prompted `Image path [test_image]:` enter the file name with or without an extension*
- The image can use any colour format but RGBA is best for accuracy
- The image can have a transparent background which becomes black in the final svg**

_*If no extension is present and two files share the same name the extensions are prioritised as follows: .jpeg > .jpg > .png_

_**A pixel with alpha > 0 will be treated as fully transparent_
### Other Parameters
- `Number of colours [2]:` Determines the number of colours in the output
- `Minimum contour area (px²) [5]:` Determines the maximum area of contours that will be present in the output
- `Maximum bridge contour area (px²) [100]:` Determines the maximum area of contours that will be bridged (set to 0 for no bridges)
- `Maximum contour distance (center to center) (px) [30]:` Determines the distance between contours for them to be bridged
- `Bridge Width (px) [1]:` Determines the width of the bridges

## Installation
### Dependencies
- NumPy
- Pillow
- scikit-learn
- opencv-python

### Environment Setup
To set up the environment for this project do the following:
#### macOS
1. Install Miniforge3 — download the installer for your Mac:
    - [Apple Silicon (M1/M2/M3)](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)
    - [Intel](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh)
2. Install Potrace using Homebrew:
    ```bash
    brew install potrace
    ```
    If you don't have Homebrew, install it first at [brew.sh](https://brew.sh)
3. Clone this repository:
    ```bash
    git clone https://github.com/lachlan-wilson/Image-to-SVG-converter.git
    cd Image-to-SVG-converter
    ```
4. Create and activate a Conda environment:
    ```bash
    conda create -n image-to-svg python=3.11
    conda activate image-to-svg
    ```
5. Install dependencies:
    ```bash
    conda install numpy pillow scikit-learn opencv
    ```
6. Run the program:
    ```bash
    python main.py
    ```

## File Structure
```
Image-to-SVG-converter/
├── main.py                                    # The python code itself
├── requirements.txt                           # Python dependencies
├── README.md                                  # Project documentation
├── examples/                                  # Folder to store example images
│   └── boat.jpg
│   └── boat_combined.jpg
├── images/                                    # Folder to store images
│   └── test_image.jpg                         # Basic image that can be used to test the program
└── test_image_output/                         # Output folder
    ├── Quantised_test_image.jpg               # Quantised image
    ├── test_image_combined.svg                # Final SVG
    ├── PNG_cleaned_layers_folder/             
    │   └── test_image_bridged_layer_1.png     # Layer 1
    │       ...
    └── PNG_layers_folder/
    │   └── test_image_layer_1.png             # Layer 1
    │       ...
    └── SVG_layers_folder/
        └── test_image_layer_1.png             # Layer 1
            ...
```
## License
This project is licensed under the [MIT License](LICENSE).
## Notes
This is hopefully my first of many projects. I am more than open to feedback/tips from anyone.

# Image to SVG converter
## Overview
This program was designed to easily turn images into SVGs that could then be used to make paper cut posters. It takes an image and converts it into an SVG. It first turns the image into layers of the most common colours, then it cleans these layers, removing small countours (groups of pixels) and bridging slighly larger ones. Using Potrace it then turns each layer into an SVG. The layers are then combined into one SVG.
## Usage
### Enputting an Image
- The image must be within the `Images` folder
- The image must be one of the following: `.jpeg`, `.jpg`, `.png`
- When prompted `Image path [test_image]:` enter the file name with or without an extension*
- The image can use any colour format but RGBA is best for accuracy
- The image can have a transparent background which becomes black in the final svg**

_*If no extension is present and two files share the same name the extensions are prioritised as follows: .jpeg > .jpg > .png_

_**A pixel with aplha > 0 will be treated as fully transparent_
### Other Parameters
- `Number of colours [2]:` Determines the number of colours in the output
- `Minimum contour area (px²) [5]:` Determines the maximum area of contours that will be present in the output
- `Maximum bridge contour area (px²) [100]:` Determines the maximum area of contours that will be bridged (set to 0 for no bridges)
- `Maximum contour distance (center to center) (px) [30]:` Determines the distance between contours for them to be bridged
- `Bridge Width (px) [1]:` Detemines the width of the bridges

## Installation
### Dependecies
- Numpy
- Pillow
- scikit-learn
- opencv-python
### Enviroment
I used a Conda enviroment through miniforge3.
## Usage


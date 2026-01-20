# Image to SVG converter
## Overview
This program takes an image (currently .jpg) and converts it into an SVG. It first turns the image into layers of the most common colours, then it cleans these layers, removing small countours (groups of pixels) and bridging slighly larger ones. Using Potrace it then turns each layer into an SVG. The layers are then combined in XML.
## Installation
### Dependecies
- Numpy
- Pillow
- scikit-learn
- opencv-python

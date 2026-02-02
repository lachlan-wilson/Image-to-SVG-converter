#   ----- Image To SVG Program -----
#        ----- 29/10/2025 -----
import os
import tempfile
import subprocess
from PIL import Image
import cv2
import numpy
from sklearn.cluster import MiniBatchKMeans
import time


# Turns a string into a title with a constant width
def title(title):
    char_length = int((50 - len(title))/2)  # Makes a constant width no matter the length of the title
    if len(title) % 2 == 1:     # Accounts for half spaces
        offset = 1
    else:
        offset = 0

    # Displays the title in blue with correct length
    print("\033[94m<" + "-" * char_length + f" {title} " + "-" * (char_length+offset) + ">\033[0m")


# Does the same thing as title but shorter and green
def subtitle(title):
    char_length = int((30 - len(title))/2)
    if len(title) % 2 == 1:
        offset = 1
    else:
        offset = 0
    print("\033[32m<" + "-" * char_length + f" {title} " + "-" * (char_length+offset) + ">\033[0m")


# Finds the squared distance between two coordinates
def get_distance(coord1, coord2):
    distance = (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2
    return distance


# Gets validated inputs
def get_inputs():
    title("Inputs")

    # Sets the default values for easier changing
    defaults = ["test_image", 2, "n", 5, 100, 30, 1]

    # Gets Image Path
    image_path = input(f"Image path (.jpg) [{defaults[0]}]: ") or defaults[0]
    # Ensures the path exists
    while not os.path.exists(f"{os.path.join('Images', image_path)}.jpg"):
        print("\033[91mError.File not found. Please enter a valid file path.\033[0m")
        image_path = input(f"Image path [{defaults[0]}]: ") or defaults[0]

    # Gets Colour Depth
    while True:
        try:
            colour_depth = int(input(f"Number of colours [{defaults[1]}]: ") or defaults[1])
            break
        except ValueError:
            print("\033[91mError. Invalid data type. Please enter an integer.\033[0m")
    while not (1 <= colour_depth <= 100):
        print("\033[91mError. Invalid integer. Please enter an integer between 1 and 100\033[0m")
        while True:
            try:
                colour_depth = int(input(f"Number of colours [{defaults[1]}]: ") or defaults[1])
                break
            except ValueError:
                print("\033[91mError. Invalid file type. Please enter an integer.\033[0m")

    # Gets whether the image has a green background.
    answer = input(f"Green background (y/n) [{defaults[2]}]: ") or defaults[2]
    while not(answer == "y" or answer == "n"):
        print('\033[91mError. Invalid input. Please enter "y" or "n" \033[0m')
        answer = input(f"Green background (y/n) [{defaults[2]}]: ") or defaults[2]
    green_background = True if answer == "y" else False

    # Gets Minimum contour Area
    while True:
        try:
            min_contour_area = int(input(f"Minimum contour area (mm\u00b2) [{defaults[3]}]: ") or defaults[3])
            while min_contour_area < 0:
                print("\033[91mError. Invalid real number. Please enter a positive real number.\033[0m")
                min_contour_area = int(input(f"Minimum contour area (mm\u00b2) [{defaults[3]}]: ") or defaults[3])
            break
        except ValueError:
            print("\033[91mError. Invalid data type. Please enter a real number.\033[0m")

    # Gets Maximum Bridge contour Area
    while True:
        try:
            max_bridge_contour_area = int(input(f"Maximum bridge contour area (mm\u00b2) [{defaults[4]}]: ") or defaults[4])
            while max_bridge_contour_area < 0:
                print("\033[91mError. Invalid real number. Please enter a positive real number.\033[0m")
                max_bridge_contour_area = int(input(f"Maximum bridge contour area (mm\u00b2) [{defaults[4]}]: ") or defaults[4])
            break
        except ValueError:
            print("\033[91mError. Invalid data type. Please enter a real number.\033[0m")

    # Gets Maximum contour Distance
    while True:
        try:
            max_contour_distance = int(input(f"Maximum contour distance (center to center) (mm) [{defaults[5]}]: ") or defaults[5])
            while max_contour_distance < 0:
                print("\033[91mError. Invalid real number. Please enter a positive real number.\033[0m")
                max_contour_distance = int(input(f"Maximum contour distance (center to center) (mm) [{defaults[5]}]: ") or defaults[5])
            break
        except ValueError:
            print("\033[91mError. Invalid data type. Please enter a real number.\033[0m")

    # Gets Bridge Width
    while True:
        try:
            bridge_width = float(input(f"Bridge Width (mm) [{defaults[6]}]: ") or defaults[6])
            while bridge_width <= 0:
                print("\033[91mError. Invalid real number. Please enter a real number above 0.\033[0m")
                bridge_width = int(input(f"Bridge Width (mm) [{defaults[6]}]: ") or defaults[6])
            break
        except ValueError:
            print("\033[91mError. Invalid data type. Please enter a real number.\033[0m")

    print("")   # Skips a line because last line could be anything

    return image_path, colour_depth, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width, green_background


# Loads image in the RBG colour space
def load_image(image_path):
    title("Loading Image")

    print("Loading image...")
    # Stores image as a numpy array
    image = cv2.imread(f"{os.path.join('Images', image_path)}.jpg")
    print("Loaded Image.\n")

    print("Creating output folder...")
    # Creates a folder with the same name as the image
    output_path = f"{image_path}_output"
    os.makedirs(output_path, exist_ok=True)
    print("Created output folder.\n")

    print("Converting image to RBG...")
    # Converts image to RGB colour space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Converted image to RGB.\n")

    return image, output_path


# Scales parameters
def scale_parameters(image, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width):

    height, width = image.shape[:2]     # Gets the height and width of the image in pixels

    length_enlargement_scale_factor = height / 420  # is used to conver mm to pixels, A3 is 420mm high
    area_enlargement_scale_factor = length_enlargement_scale_factor ** 2

    # Scales distances and areas by the correct factors
    max_bridge_contour_area = int(max_bridge_contour_area * area_enlargement_scale_factor)
    min_contour_area = int(min_contour_area * area_enlargement_scale_factor)
    max_contour_distance = int(max_contour_distance * length_enlargement_scale_factor)
    bridge_width = int(bridge_width * length_enlargement_scale_factor)

    return max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width


# Quantises the image (Turns it into a few colours only)
def quantise_image(image, colour_depth, image_path, output_path, green_background):
    title("Quantising Image")

    print("Converting image to LAB...")
    # Converts the image to LAB colour space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    print("Converted image to LAB.\n")

    print("Reshaping image...")
    height, width = image.shape[:2]     # Gets the height and width of the image in pixels
    n_pixels = height * width
    # Flattens array so each pixel has an index with 3 colours, int16 allows for calculating distances better
    image = image.reshape((-1, 3)).astype(numpy.int16)
    print("Reshaped image.\n")

    # If the image has a green background AKA should have a transparent background
    if green_background:
        print("Removing green background pixels...")
        # Defines the colour in LAB colour spaces and correct encoding
        green_colour = numpy.array([227, 57, 210], dtype=numpy.int16)
        green_tolerance = 10    # Adds a tolerance to account for changing encoding
        # Creates an array holding the difference between each pixel in the image and the green colour
        diff = numpy.abs(image - green_colour)
        # Creates a boolean array that holds the green pixels
        background = numpy.all(diff <= green_tolerance, axis=1)
        n_bg_pixels = int(numpy.sum(background))    # Counts the pixels in the array
        image = image[~background]    # Removes what were the green pixels
        print(f"Removed {n_bg_pixels} green background pixels.\n")

        # Creates a tuple containing all the colours
        print(f"Forming {colour_depth} colours...")
        os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Means it only uses one CPU core?
        # Finds the most important colours in the image excluding the background
        kmeans = MiniBatchKMeans(n_clusters=(colour_depth-1), random_state=42, batch_size=2048, )
        pixel_labels_no_bg = kmeans.fit_predict(image)  # Labels each pixel with its colour
        colour_groups = kmeans.cluster_centers_
        print(f"Formed {colour_depth} colours.\n")

        print(f"Adding background...")
        # Creates an array the same size as the original image where each pixel has index -1 (for background)
        pixel_labels = numpy.full((n_pixels,), fill_value=(colour_depth-1), dtype=int)
        # Adds the correct pixel labels to the background where the green pixels where not present
        pixel_labels[~background] = pixel_labels_no_bg
        black = numpy.array([0, 128, 128], dtype=numpy.float64)     # Creates a black in LAB
        colour_groups = numpy.vstack([colour_groups, black])    # Adds the black colour to the others
        print(f"Added background.")

    else:
        # Creates a tuple containing all the colours
        print(f"Forming {colour_depth} colours...")
        os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Means it only uses one CPU core?
        # Finds the most important colours in the image
        kmeans = MiniBatchKMeans(n_clusters=colour_depth, random_state=42, batch_size=2048, )
        pixel_labels = kmeans.fit_predict(image)    # Labels each pixel with its colour
        colour_groups = kmeans.cluster_centers_
        print(f"Formed {colour_depth} colours.\n")

    # Rebuilds the labels and colour groups to be sorted by lightness
    print("Sorting colours by lightness...")
    colour_groups_order = (numpy.argsort(colour_groups[:, 0]))  # Order of the colours based on the L channel
    colour_groups = colour_groups[colour_groups_order]  # Reorders the actual array
    pixel_label_map = numpy.zeros_like(colour_groups_order)     # Creates a blank array the right size
    pixel_label_map[colour_groups_order] = numpy.arange(len(colour_groups_order))   # Creates an order for the labels
    pixel_labels = pixel_label_map[pixel_labels]    # Applies the new order
    print("Sorted colours by lightness.\n")

    # Rebuilds the image for use in the next step and to be saved
    print("Rebuilding quantised image...")
    image = colour_groups[pixel_labels].astype("uint8")     # Applies the labels and order
    image = image.reshape((height, width, 3))   # Reshapes the image to it cv2 can work with it
    print("Rebuilt quantised image.\n")

    print("Converting image to RBG...")
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)  # Changes the colour back to RGB
    print("Converted image to RGB.\n")

    # Writes the image to the output folder with an appropriate name
    print("Saving quantised image...")
    cv2.imwrite(f"{output_path}/Quantised_{image_path}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print("Saved quantised image.\n")

    return image, colour_groups, pixel_labels.reshape(height, width), colour_groups_order, height, width


# Turns the image into binary (black and white) layers
def build_binary_layers(pixel_labels, image_path, output_path, colour_depth):
    title("Building Binary Layers")

    # Creates a folder for the layers to go into
    print("Creating output folder...")
    path = os.path.join(output_path, "PNG_layers_folder")
    os.makedirs(path, exist_ok=True)
    print("Created output folder.\n")

    print("Building & saving layers...\n")
    layers = []     # Initialises an empty array for the layers to go into
    for i in range(colour_depth):   # Loops for each colour
        subtitle(f"Layer {i+1}")

        print(f"Building layer {i + 1}...")
        # Creates a layer with only the pixels in the relevant colour
        layer = numpy.isin(pixel_labels, numpy.arange(i, colour_depth))
        layer = (layer.astype(numpy.uint8) * 255)   # Makes these pixels white (for binary layers)
        layer = cv2.bitwise_not(layer)  # Inverts the layer
        layers.append(layer)    # Adds the layer to the layers array
        print(f"Built layer {i+1}.\n")

        # Writes the layer to the output folder
        print(f"Saving layer {i+1}...")
        filename = os.path.join(path, f"{image_path}_layer_{i+1}.png")
        cv2.imwrite(filename, layer)
        print(f"Saved layer {i + 1}.\n")
    print("Build & saved layers.\n")
    return layers


# Gets rid of small contours, bridges medium-sized ones and deals with diagonal joins
def clean_contours(layers, image_path, output_path, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width):
    title("Cleaning contours")

    # Creates a folder for the layers to go into
    print("Creating output folder...")
    path = os.path.join(output_path, "PNG_cleaned_layers_folder")
    os.makedirs(path, exist_ok=True)
    print("Created output folder.\n")

    print("Cleaning all contours...\n")
    total_contours = 0  # Initialises a counter
    bridged_layers = []     # Initialises an empty array for the layers to go into
    for i, layer in enumerate(layers):  # Loops for each layer
        subtitle(f"Layer {i + 1}")
        print(f"Cleaning contours in layer {i + 1}...\n")
        layer = cv2.bitwise_not(layer)  # Inverts the layer

        print("Finding contours...")
        # Finds the contours(edges) of the binary layers and stores whether they're holes or fills
        contours, hierarchy = cv2.findContours(layer, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if len(contours) <= 1:  # If it's one big contour
            bridged_layers.append(layer)    # Adds the layer to the layers array
            total_contours += 1     # Counts the contour
            print(f"Skipped layer {i + 1} as there was only one contour.\n")
            # Writes the layer to the folder
            filename = os.path.join(path, f"{image_path}_cleaned_layer_{i + 1}.png")
            cv2.imwrite(filename, cv2.bitwise_not(layer))
            continue    # Skips to the next interation of the loop

        n_of_contours_before = 0    # Initialises a counter
        hierarchy = hierarchy[0]    # Flattens the hierarchy array

        for hier in hierarchy:  # Loop for each contour
            if hier[3] == -1:   # If fill
                n_of_contours_before += 1   # Counts the contour
        print(f"Found {n_of_contours_before} contours.\n")

        print("Categorising contours...")
        # Initialises arrays to store contours in based on their nature
        fills = []
        holes = []
        small = []
        bridged_fills = []

        for contour, hier in zip(contours, hierarchy):  # Loop for each contour
            if hier[3] == -1:   # If the contour is filled
                area = cv2.contourArea(contour)     # Finds the area of the contour
                if area > min_contour_area:     # If the contour is big enough
                    fills.append(contour)   # Add the contour to the fills array
                    if area < max_bridge_contour_area:  # If the contour is small enough to be bridged
                        bridged_fills.append(contour)   # Add the contour to the bridged_fills array
                else:
                    small.append(contour)   # Add the contour to the small array
            else:
                holes.append(contour)   # Add the contour to the holes array
        print(f"Categorised contours. Found {len(bridged_fills)} contours suitable for bridging.\n")

        print("Culling small contours...")
        cv2.drawContours(layer, small, -1, color=0, thickness=-1)   # Removes the small contours from the layer
        print(f"Culled {n_of_contours_before - len(fills)} small contours.\n")

        print("Finding centres of all contours...")
        # Finds the contours of the updated layer
        contours, hierarchy = cv2.findContours(layer, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour_centers = []    # Initialises an array to store the contour's centers

        for contour in contours:
            # Finds the centre and stores it as a tuple of coordinates
            m = cv2.moments(contour)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                contour_centers.append((cx, cy))
            else:
                contour_centers.append((None, None))
        print("Found centres of all contours.\n")

        # Finds the centres of bridged contours only
        print("Finding centres of bridge contours...")
        bridged_fills_centers = []
        for bridged_fill in bridged_fills:
            m = cv2.moments(bridged_fill)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                bridged_fills_centers.append((cx, cy))
            else:
                bridged_fills_centers.append((None, None))
        print("Found centres of bridge contours.\n")

        # Checks if a bridge is eligible and if so draws a bridge
        print("Adding bridges...")
        n_bridges = 0   # Initialises a counter
        # Loops for each bridged fill and adds an incremental index
        for i2, (bridged_fill, bridged_fill_center) in enumerate(zip(bridged_fills, bridged_fills_centers)):
            best_distance = float("inf")    # Initialises the best distance as infinity

            for contour, contour_center in zip(contours, contour_centers):  # Check against every other contour
                if contour_center == (None, None):  # Skip if no centre was found
                    continue
                distance = get_distance(bridged_fill_center, contour_center)    # finds the distance between the centers

                # If this is the new lowest distance that's not the distance between itself
                if distance < best_distance and distance != 0:
                    nearest_contour = contour   # Update the best contour
                    best_distance = distance    # Update the lowest distance

            if best_distance < max_contour_distance**2:     # If the best distance is less than the max distance (squared to optimise)

                best_distance = float("inf")    # Initialises a new best distance for each point as infinity
                for test_coord1 in bridged_fill:    # For each point in the bridged contour
                    for test_coord2 in nearest_contour:     # For each point in the nearest contour
                        distance = get_distance(test_coord1[0], test_coord2[0])     # Finds the distance between the two points

                        if distance < best_distance:    # If it's the new lowest update the coordinates
                            best_distance = distance
                            coord1 = (test_coord1[0])
                            coord2 = (test_coord2[0])

                cv2.line(layer, coord1, coord2, color=255, thickness=bridge_width)  # Draw a line between the two coordinates
                n_bridges += 1  # Count that bridge
                # DIsplays a message with stats
            print(f"\rContour: {i2}/{len(bridged_fills)} - {int(round((i2 / len(bridged_fills)) * 100, 0))}%", end="")
        print(f"\nAdded {n_bridges} bridges.\n")

        # Fills diagonals because cv2 counts them as connected by Potrace doesn't (needs improved)
        print("Filling diagonals...")
        total_changes = 0
        while True:
            img = (layer == 255).astype(numpy.uint8)

            # Extract shifted views (top-left 2×2 windows)
            a = img[:-1, :-1]
            b = img[:-1, 1:]
            c = img[1:, :-1]
            d = img[1:, 1:]

            # pattern1: 0 1 / 1 0  -> fill a and d
            pattern1 = (a == 0) & (b == 1) & (c == 1) & (d == 0)

            # pattern2: 1 0 / 0 1  -> fill b and c
            pattern2 = (a == 1) & (b == 0) & (c == 0) & (d == 1)

            # Combine the masks (True = we must fill both diagonals)
            mask = pattern1 | pattern2

            if not numpy.any(mask):
                break

            changes = 0
            if numpy.any(pattern1):
                # set a and d to 1 where pattern1 is True
                # note: use copy-on-write semantics to avoid unintended overlap issues
                idx = numpy.nonzero(pattern1)
                img[:-1, :-1][idx] = 1
                img[1:, 1:][idx] = 1
                changes += idx[0].size * 2  # two pixels per match
            if numpy.any(pattern2):
                idx = numpy.nonzero(pattern2)
                img[:-1, 1:][idx] = 1
                img[1:, :-1][idx] = 1
                changes += idx[0].size * 2

            total_changes += changes
            # Create a copy to modify
            fixed = img.copy()

            # Fill the diagonal pixels
            # For pattern1 and pattern2, fill both diagonal positions
            fixed[:-1, :-1][mask] = 1
            fixed[1:, 1:][mask] = 1

            # Convert back to 0/255 format
            layer = (fixed * 255).astype(numpy.uint8)
        print(f"Filled {total_changes} diagonals.")

        print(f"Saving layer {i + 1}...")
        layer = cv2.bitwise_not(layer)  # Inverts the image
        filename = os.path.join(path, f"{image_path}_bridged_layer_{i + 1}.png")
        cv2.imwrite(filename, layer)    # Writes the image to the output folder
        print(f"Saved layer {i + 1}.\n")

        # Counts contours at the end to display stats
        print("Finding contours...")
        contours, hierarchy = cv2.findContours(cv2.bitwise_not(layer), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        n_of_contours_after = 0

        hierarchy = hierarchy[0]

        for hier in hierarchy:
            if hier[3] == -1:
                n_of_contours_after += 1

        total_contours += n_of_contours_after
        print(f"Found {n_of_contours_after} contours.")
        print(f"Removed {n_of_contours_before - n_of_contours_after} contours.\n")

        bridged_layers.append(layer)
    print("Bridged all contours.\n")
    return bridged_layers, total_contours   # Returns the layers and the total contours for stats


# Converts each layer to an SVG
def convert_layers_to_svg(layers, image_path, colour_groups, output_path, height, width, colour_depth, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width, green_background ):
    title("Converting Layers To SVGs")

    # Creates a folder for the layers to go into
    print("Creating output folder...")
    path = os.path.join(output_path, "SVG_layers_folder")
    os.makedirs(path, exist_ok=True)
    print("Created output folder.\n")

    print("Vectorising layers...\n")
    potrace_path = "/usr/local/bin/potrace"     # Absolute path of Potrace
    # Creates a temporary file that will be used and then deleted
    with tempfile.TemporaryDirectory(prefix="potrace_tmp_") as temp_dir:
        for i, layer in enumerate(layers):   # Loops for each layer
            subtitle(f"Layer {i + 1}")
            print(f"Vectorising layer {i + 1}...\n")

            print("Setting up file...")
            svg_filename = os.path.join(path, f"{image_path}_layer_{i + 1}.svg")    # Creates an appropriate filename
            print("Set up file.\n")

            if i == 0:  # If it's the bottom layer
                print("Making bottom layer a rectangle...")
                # SVG code for a rectangle that aligns with Potrace formatting
                svg_data = (f'''<?xml version="1.0" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"
     "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
    <svg version="1.0" xmlns="http://www.w3.org/2000/svg"
     width="{width}.000000pt" height="{height}.000000pt" viewBox="0 0 {width}.000000 {height}.000000"
     preserveAspectRatio="xMidYMid meet">
    <metadata>
    Create by Lachlan Wilson
    Input Pararameters: 
        colour depth: {colour_depth}
        Min contour area: {min_contour_area}
        Max bridge contour area: {max_bridge_contour_area}
        Max bridge contour distance: {max_contour_distance}
        Bridge width: {bridge_width}
    Created using potrace 1.16, written by Peter Selinger 2001-2019
    </metadata>
    <g transform="translate(0.000000,{height}.000000) scale(1,-1)"
    fill="#000000" stroke="none">
    <rect width="{width}" height="{height}" x="0" y="0" />
    </g>
    </svg>''')
                # Opens the file, writes the data and closes it
                with open(svg_filename, "w", encoding="utf-8") as f:
                    f.write(svg_data)

                print("Made bottom layer a rectangle.\n")

            else:
                print("Converting PNG to SVG using Potrace...")

                print("Setting up temporary files...")
                temp_pbm = os.path.join(temp_dir, f"mask_{i}.pbm")  # Creates an empty bitmap file in the temporary folder
                Image.fromarray(layer).convert("1").save(temp_pbm)  # Turns the array into a pbm file and saves it
                print("Set up temporary files.\n")

                arguments = [
                    potrace_path,
                    temp_pbm,
                    "-s",
                    "-o",
                    svg_filename,
                    "--alphamax",
                    "1.2",
                    "--turdsize",
                    "2"
                ]
                subprocess.run(arguments, check=True)   # Runs Potrace with the given arguments
                print("Converted PNG to SVG using Potrace.\n")

            print("Adding colours...")
            RGB_colour = cv2.cvtColor(numpy.uint8([[colour_groups[i]]]), cv2.COLOR_Lab2RGB)[0][0]   # Converts the layer's colour to RGB
            HEX_colour = ('#%02x%02x%02x' % tuple(int(c) for c in RGB_colour))  # Converts the RGB colours into RGB Hex
            print(f"Layer colour:{HEX_colour}")
            with open(svg_filename, "r", encoding="utf-8") as svg_data:
                svg_data = svg_data.read()
                svg_data = svg_data.replace('fill="#000000"', f'fill="{HEX_colour}"')   # Replaces the colour value with the correct colour
            print("Added colours.\n")

        print(f"Vectorised layer {i + 1}.\n")

        print(f"Saving layer {i+1}...")
        # Writes the data to the file
        with open(svg_filename, "w", encoding="utf-8") as f:
            f.write(svg_data)
        print(f"Saved layer {i + 1}.\n")

    print("Vectorised layers.\n")


# Combines and saves the SVG layers
def combine_svgs(image_path, output_path, colour_depth):
    title("Combining SVGs")

    print("Formatting layers...\n")
    combined_svg_code = ""  # Inititalises an empty string for the combined code
    for i in range(colour_depth):   # Loops for each colour/layer
        subtitle(f"Layer {i+1}")

        # Opens the file containing the code for that layer
        print("Opening file...")
        svg_filename = os.path.join(output_path, "SVG_layers_folder", f"{image_path}_layer_{i + 1}.svg")
        with open(svg_filename, "r", encoding="utf-8") as file:
            svg_code = file.read()
        print("Opened file.\n")

        print("Formatting code...")
        substring_index = svg_code.find("</metadata>") + 12     # Finds the index of the start of the path code
        # Takes the root (beginning) of the SVG code and stores it to be appended later
        if i == 0:
            svg_root = svg_code[:substring_index]
        svg_code = svg_code[substring_index:-6]     # Stores the actual SVG code with all the paths.
        print("Formatted code.\n")

        print("Appending code...")
        combined_svg_code += svg_code   # Appends this layers paths
        print("Appended code.\n")

    print("Formatted layers.\n")

    combined_svg_code = svg_root + combined_svg_code + "</svg>"     # Combines the paths with the root and closes the section

    combined_svg_code = combined_svg_code.replace("<<", "<")    # Fixes discrepancies

    print("Saving layers...")
    # Writes the code to the file
    svg_filename = os.path.join(output_path, f"{image_path}_combined.svg")
    with open(svg_filename, "w", encoding="utf-8") as file:
        file.write(combined_svg_code)
    print("Saved layers.\n")


# Calls the appropriate functions with the appropraite arguments
def main():
    image_path, colour_depth, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width, green_background = get_inputs()
    start = time.perf_counter()     # Starts the timer once the inputs have been given
    image, output_path = load_image(image_path)
    max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width = scale_parameters(image, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width)
    image, colour_groups, pixel_labels, colour_groups_order, height, width = quantise_image(image, colour_depth, image_path, output_path, green_background)
    layers = build_binary_layers(pixel_labels, image_path, output_path, colour_depth)
    bridged_layers, total_contours = clean_contours(layers, image_path, output_path, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width)
    convert_layers_to_svg(bridged_layers, image_path, colour_groups, output_path, height, width, colour_depth, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width, green_background)
    combine_svgs(image_path, output_path, colour_depth)

    end = time.perf_counter()   # Stops the timer and displays the results
    print(f"Time: {end - start:.6f} seconds.\n")
    print(f"Total contours: {total_contours}.\n")
    title("Complete")


main()
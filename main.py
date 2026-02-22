#   ----- Image To SVG Program -----
#        ----- 29/10/2025 -----
import os
import tempfile
import subprocess
from PIL import Image, ImageOps
import cv2
import numpy
from sklearn.cluster import MiniBatchKMeans
import time
import shutil


# Turn a string into a title with a constant width
def title(title):
    char_length = int((50 - len(title))/2)  # Make a constant width no matter the length of the title
    if len(title) % 2 == 1:     # Account for half spaces
        offset = 1
    else:
        offset = 0

    # Display the title in blue with correct length
    print("\033[94m<" + "-" * char_length + f" {title} " + "-" * (char_length+offset) + ">\033[0m")


# The same thing as title but shorter and green
def subtitle(title):
    char_length = int((30 - len(title))/2)
    if len(title) % 2 == 1:
        offset = 1
    else:
        offset = 0
    print("\033[32m<" + "-" * char_length + f" {title} " + "-" * (char_length+offset) + ">\033[0m")


# Find the squared distance between two coordinates
def get_distance(coord1, coord2):
    distance = (coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2
    return distance


# Get validated inputs
def get_inputs():
    title("Inputs")

    # Set the default values for easier changing
    defaults = ["test_image", 2, 5, 100, 30, 1]

    # Get Image Path
    image_path = input(f"Image path [{defaults[0]}]: ") or defaults[0]
    # Initialise an array with all the supported extensions
    supported_extensions = [".jpeg", ".jpg", ".png"]
    valid = False   # Initialise valid as False
    while not valid:
        if image_path[image_path.rfind("."):] in supported_extensions:  # If the path already has an extension
            if os.path.exists(f"{os.path.join('Images', image_path)}"):     # Check if it can be found
                valid = True  # Stop the next iteration of the loop
        else:
            for ext in supported_extensions:    # For each possible extension
                if os.path.exists(f"{os.path.join('Images', image_path)}{ext}"):    # Check if it can be found
                    valid = True    # Stop the next iteration of the loop
                    image_path = image_path + ext   # Stores the found image path
        if valid:   # Stop the loop if a valid path was found
            break
        # Get a new input
        print("\033[91mError. File not found. Please enter a valid file path.\033[0m")
        image_path = input(f"Image path [{defaults[0]}]: ") or defaults[0]

    # Get the colour depth
    while True: # Loop until broken
        try:    # Try this first
            colour_depth = int(input(f"Number of colours [{defaults[1]}]: ") or defaults[1])    # Get the input
            break   # If this suceeds break from the loop
        except ValueError:  # If this crashed
            print("\033[91mError. Invalid data type. Please enter an integer.\033[0m")  # Display an error message

    while not (1 <= colour_depth <= 100):   # Loop while the colour depth is not a valid integer
        print("\033[91mError. Invalid integer. Please enter an integer between 1 and 100\033[0m")   # Display an error message
        # Get a new input
        while True:
            try:
                colour_depth = int(input(f"Number of colours [{defaults[1]}]: ") or defaults[1])
                break
            except ValueError:
                print("\033[91mError. Invalid file type. Please enter an integer.\033[0m")

    # Get the minimum contour Area
    while True:
        try:
            min_contour_area = int(input(f"Minimum contour area (px\u00b2) [{defaults[2]}]: ") or defaults[2])
            while min_contour_area < 0:
                print("\033[91mError. Invalid real number. Please enter a positive real number.\033[0m")
                min_contour_area = int(input(f"Minimum contour area (px\u00b2) [{defaults[2]}]: ") or defaults[2])
            break
        except ValueError:
            print("\033[91mError. Invalid data type. Please enter a real number.\033[0m")

    # Get the maximum bridge contour Area
    while True:
        try:
            max_bridge_contour_area = int(input(f"Maximum bridge contour area (px\u00b2) [{defaults[3]}]: ") or defaults[3])
            while max_bridge_contour_area < 0:
                print("\033[91mError. Invalid real number. Please enter a positive real number.\033[0m")
                max_bridge_contour_area = int(input(f"Maximum bridge contour area (px\u00b2) [{defaults[3]}]: ") or defaults[3])
            break
        except ValueError:
            print("\033[91mError. Invalid data type. Please enter a real number.\033[0m")

    # Get the maximum contour distance
    while True:
        try:
            max_contour_distance = int(input(f"Maximum contour distance (center to center) (px) [{defaults[4]}]: ") or defaults[4])
            while max_contour_distance < 0:
                print("\033[91mError. Invalid real number. Please enter a positive real number.\033[0m")
                max_contour_distance = int(input(f"Maximum contour distance (center to center) (px) [{defaults[4]}]: ") or defaults[4])
            break
        except ValueError:
            print("\033[91mError. Invalid data type. Please enter a real number.\033[0m")

    # Get the bridge width
    while True:
        try:
            bridge_width = float(input(f"Bridge Width (px) [{defaults[5]}]: ") or defaults[5])
            while bridge_width <= 0:
                print("\033[91mError. Invalid real number. Please enter a real number above 0.\033[0m")
                bridge_width = int(input(f"Bridge Width (px) [{defaults[5]}]: ") or defaults[5])
            break
        except ValueError:
            print("\033[91mError. Invalid data type. Please enter a real number.\033[0m")

    print("")   # Skip a line because last line could be anything

    return image_path, colour_depth, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width


# Load the image in the RGBA colour space
def load_image(image_path):
    title("Loading Image")

    print("Loading image...")
    image = Image.open(os.path.join('Images', image_path))  # Open image
    image = ImageOps.exif_transpose(image)  # Ensures image is correctly orientated
    image = image.convert("RGBA")   # Convert the image to RGBA
    image = numpy.array(image)  # Conver the image to a numpy array of pixels
    print("Loaded Image.\n")

    print("Creating output folder...")
    # Creates a folder with the same name as the image
    output_path = f"{image_path[:image_path.rfind('.')]}_output"

    # If the folder already exists delete it
    if os.path.exists(output_path):
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)

    os.makedirs(output_path, exist_ok = False) # Make the folder
    print("Created output folder.\n")

    return image, output_path


# Quantise the image (Turn it into a few colours only)
def quantise_image(image, colour_depth, image_path, output_path):
    title("Quantising Image")

    print("Selecting transparent pixels...")
    alpha = image[..., 3]   # Array of the alpha channel
    background = (alpha < 255).reshape(-1)  # Boolean array of transparent pixels
    print("Selected transparent pixels.")

    print("Converting image to LAB...")
    image = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2LAB) # Convert the image to LAB colour space
    print("Converted image to LAB.\n")

    print("Reshaping image...")
    height, width = image.shape[:2]     # Get the height and width of the image in pixels
    n_pixels = height * width
    # Flatten the array so each pixel has an index with 3 colours, int16 allows for calculating distances better
    image = image.reshape((-1, 3)).astype(numpy.int16)
    print("Reshaped image.\n")

    print("Removing background pixels...")
    n_bg_pixels = int(numpy.sum(background))    # Count the pixels in the array
    colour_depth_offset = -1 if n_bg_pixels > 0 else 0  # If the image has transparent pixels remove a colour
    image = image[~background]    # Remove what were the transparent pixels
    print(f"Removed {n_bg_pixels} background pixels.\n")

    # Create a tuple containing all the colours and an array referencing those colours
    print(f"Forming {colour_depth} colours...")
    os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Only use one CPU core?
    # Find the most important colours in the image
    kmeans = MiniBatchKMeans(n_clusters=(colour_depth + colour_depth_offset), random_state=42, batch_size=2048, )
    pixel_labels = kmeans.fit_predict(image)  # Label each pixel with its colour
    colour_groups = kmeans.cluster_centers_     # Create a tuple to store the colours
    print(f"Formed {colour_depth} colours.\n")

    if n_bg_pixels > 0: # If the image has transparent pixels
        print(f"Adding background...")
        # Create an array the same size as the original image where each pixel has an index of -1 (for the background)
        pixel_labels = numpy.full((n_pixels,), fill_value=(colour_depth + colour_depth_offset), dtype=int)
        # Add the correct pixel labels to the background where the transparent pixels where not present
        pixel_labels[~background] = pixel_labels
        black = numpy.array([0, 128, 128], dtype=numpy.float64)     # Black in LAB
        colour_groups = numpy.vstack([colour_groups, black])    # Add the black colour to the others
        print(f"Added background.")

    # Rebuild the labels and colour groups to be sorted by lightness
    print("Sorting colours by lightness...")
    colour_groups_order = (numpy.argsort(colour_groups[:, 0]))  # Order of the colours based on the L channel
    colour_groups = colour_groups[colour_groups_order]  # Reorder the actual array
    pixel_label_map = numpy.zeros_like(colour_groups_order)     # Create a blank array the right size
    pixel_label_map[colour_groups_order] = numpy.arange(len(colour_groups_order))   # Create an order for the labels
    pixel_labels = pixel_label_map[pixel_labels]    # Apply the new order
    print("Sorted colours by lightness.\n")

    # Rebuild the image for use in the next step and to be saved
    print("Rebuilding quantised image...")
    image = colour_groups[pixel_labels].astype("uint8")     # Apply the labels and order
    image = image.reshape((height, width, 3))   # Reshape the image so that cv2 can work with it
    print("Rebuilt quantised image.\n")

    print("Converting image to RBG...")
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)  # Change the colour back to RGB
    print("Converted image to RGB.\n")

    # Write the image to the output folder with an appropriate name
    print("Saving quantised image...")
    cv2.imwrite(f"{output_path}/Quantised_{image_path}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print("Saved quantised image.\n")

    return image, colour_groups, pixel_labels.reshape(height, width), colour_groups_order, height, width


# Turn the image into binary (black and white) layers
def build_binary_layers(pixel_labels, image_path, output_path, colour_depth):
    title("Building Binary Layers")

    # Create a folder for the layers to go into
    print("Creating output folder...")
    path = os.path.join(output_path, "PNG_layers_folder")
    os.makedirs(path, exist_ok=True)
    print("Created output folder.\n")

    print("Building & saving layers...\n")
    layers = []     # Initialise an empty array for the layers to go into

    layer = (pixel_labels >= 0).astype(numpy.uint8) * 255 # A full array of the correct size, same as the bottom layer.
    layer = cv2.bitwise_not(layer)  # Invert the layer

    for i in range(colour_depth):   # Loop for each colour
        subtitle(f"Layer {i+1}")

        print(f"Building layer {i + 1}...")
        if i > 0:   # If it's not the bottom layer
            layer[pixel_labels == i - 1] = 255  # Remove the colour below this one

        layers.append(layer.copy())    # Add the layer to the layers array, layer.copy() is used because of the way arrays are stored in memory
        print(f"Built layer {i+1}.\n")

        # Write the layer to the output folder
        print(f"Saving layer {i+1}...")
        filename = os.path.join(path, f"{image_path}_layer_{i+1}.png")
        cv2.imwrite(filename, layer)
        print(f"Saved layer {i + 1}.\n")
    print("Build & saved layers.\n")
    return layers


# Get rid of small contours, bridge medium-sized ones and deal with diagonal joins
def clean_contours(layers, image_path, output_path, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width):
    title("Cleaning contours")

    # Create a folder for the layers to go into
    print("Creating output folder...")
    path = os.path.join(output_path, "PNG_cleaned_layers_folder")
    os.makedirs(path, exist_ok=True)
    print("Created output folder.\n")

    print("Cleaning all contours...\n")
    total_contours = 0  # Initialise a counter
    bridged_layers = []     # Initialise an empty array for the layers to go into
    for i, layer in enumerate(layers):  # Loop for each layer
        subtitle(f"Layer {i + 1}")
        print(f"Cleaning contours in layer {i + 1}...\n")
        layer = cv2.bitwise_not(layer)  # Invert the layer

        print("Finding contours...")
        # Find the contours(edges) of the binary layers and stores whether they're holes or fills
        contours, hierarchy = cv2.findContours(layer, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if len(contours) <= 1:  # If it's one big contour
            bridged_layers.append(layer)    # Add the layer to the layers array
            total_contours += 1     # Count the contour
            print(f"Skipped layer {i + 1} as there was only one contour.\n")
            # Write the layer to the folder
            filename = os.path.join(path, f"{image_path}_cleaned_layer_{i + 1}.png")
            cv2.imwrite(filename, cv2.bitwise_not(layer))
            continue    # Skip to the next interation of the loop

        n_of_contours_before = 0    # Initialise a counter
        hierarchy = hierarchy[0]    # Flatten the hierarchy array

        for hier in hierarchy:  # Loop for each contour
            if hier[3] == -1:   # If the contour is a fill
                n_of_contours_before += 1   # Count the contour
        print(f"Found {n_of_contours_before} contours.\n")

        print("Categorising contours...")
        # Initialise arrays to store contours in based on their nature
        fills = []
        holes = []
        small = []
        bridged_fills = []

        for contour, hier in zip(contours, hierarchy):  # Loop for each contour
            if hier[3] == -1:   # If the contour is filled
                area = cv2.contourArea(contour)     # Find the area of the contour
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
        cv2.drawContours(layer, small, -1, color=0, thickness=-1)   # Remove the small contours from the layer
        print(f"Culled {n_of_contours_before - len(fills)} small contours.\n")

        print("Finding centres of all contours...")
        # Find the contours of the updated layer
        contours, hierarchy = cv2.findContours(layer, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour_centers = []    # Initialise an array to store the contour's centers

        for contour in contours:    # Loop for every contour
            # Find the centre and store it as a tuple of coordinates
            m = cv2.moments(contour)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                contour_centers.append((cx, cy))
            else:
                contour_centers.append((None, None))
        print("Found centres of all contours.\n")

        # Find the centres of bridged contours only
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

        # Check if a bridge is eligible and if so draw a bridge
        print("Adding bridges...")
        n_bridges = 0   # Initialise a counter
        # Loop for each bridged fill and adds an incremental index
        for i2, (bridged_fill, bridged_fill_center) in enumerate(zip(bridged_fills, bridged_fills_centers)):
            best_distance = float("inf")    # Initialise the best distance as infinity

            for contour, contour_center in zip(contours, contour_centers):  # Check against every other contour
                if contour_center == (None, None):  # Skip if no centre was found
                    continue
                distance = get_distance(bridged_fill_center, contour_center)    # Find the distance between the centers

                # If this is the new lowest distance that's not the distance between itself
                if distance < best_distance and distance != 0:
                    nearest_contour = contour   # Update the best contour
                    best_distance = distance    # Update the lowest distance

            if best_distance < max_contour_distance**2:     # If the best distance is less than the max distance (squared to optimise)

                best_distance = float("inf")    # Initialise a new best distance for each point as infinity
                for test_coord1 in bridged_fill:    # For each point in the bridged contour
                    for test_coord2 in nearest_contour:     # For each point in the nearest contour
                        distance = get_distance(test_coord1[0], test_coord2[0])     # Find the distance between the two points

                        if distance < best_distance:    # If it's the new lowest update the coordinates
                            best_distance = distance
                            coord1 = (test_coord1[0])
                            coord2 = (test_coord2[0])

                cv2.line(layer, coord1, coord2, color=255, thickness=bridge_width)  # Draw a line between the two coordinates
                n_bridges += 1  # Count that bridge
                # Display a message with stats
            print(f"\rContour: {i2}/{len(bridged_fills)} - {int(round((i2 / len(bridged_fills)) * 100, 0))}%", end="")
        print(f"\nAdded {n_bridges} bridges.\n")

        # Fill diagonals because cv2 counts them as connected by Potrace doesn't
        print("Filling diagonals...")
        layer_binary = (layer == 255).astype(numpy.uint8)   # A new array the same as layer but where a pixel was 255 it is now 1

        # Parallel arrays of every 2x2 square of pixels
        a = layer_binary[:-1, :-1]
        b = layer_binary[:-1, 1:]
        c = layer_binary[1:, :-1]
        d = layer_binary[1:, 1:]

        # An array the same size as the image but -1 in height and width because it's for each 2x2 square, if the square is a diagonal it's set to True
        pattern1 = (a == 0) & (b == 1) & (c == 1) & (d == 0)

        # Same but for the second diagonal pattern
        pattern2 = (a == 1) & (b == 0) & (c == 0) & (d == 1)

        mask = pattern1 | pattern2  # An array the same size as a patter but True if either of the patterns are True

        # Sets the pixels that are in any 2x2 square that was diagonal to be filled
        layer_binary[:-1, :-1][mask] = 1
        layer_binary[:-1, 1:][mask] = 1
        layer_binary[1:, :-1][mask] = 1
        layer_binary[1:, 1:][mask] = 1

        total_changes = int(numpy.sum(mask))    # The number of Trues in the mask
        layer = (layer_binary * 255).astype(numpy.uint8)    # Change it back to an array of 0s and 255s
        print(f"Filled {total_changes} diagonals.")

        print(f"Saving layer {i + 1}...")
        layer = cv2.bitwise_not(layer)  # Invert the image
        filename = os.path.join(path, f"{image_path}_bridged_layer_{i + 1}.png")
        cv2.imwrite(filename, layer)    # Write the image to the output folder
        print(f"Saved layer {i + 1}.\n")

        # Count contours at the end to display stats
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

        bridged_layers.append(layer)    # Add the layer to the total layers to be transferred to the next function
    print("Bridged all contours.\n")
    return bridged_layers, total_contours   # Return the layers and the total contours for stats


# Convert each layer to an SVG
def convert_layers_to_svg(layers, image_path, colour_groups, output_path, height, width, colour_depth, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width,):
    title("Converting Layers To SVGs")

    # Create a folder for the layers to go into
    print("Creating output folder...")
    path = os.path.join(output_path, "SVG_layers_folder")
    os.makedirs(path, exist_ok=True)
    print("Created output folder.\n")

    print("Vectorising layers...\n")
    potrace_path = "/usr/local/bin/potrace"     # Absolute path of Potrace
    # Create a temporary file that will be used and then deleted
    with tempfile.TemporaryDirectory(prefix="potrace_tmp_") as temp_dir:
        for i, layer in enumerate(layers):   # Loop for each layer
            subtitle(f"Layer {i + 1}")
            print(f"Vectorising layer {i + 1}...\n")

            print("Setting up file...")
            svg_filename = os.path.join(path, f"{image_path}_layer_{i + 1}.svg")    # Create an appropriate filename
            print("Set up file.\n")

            if i == 0:  # If it's the bottom layer
                print("Making bottom layer a rectangle...")
                # SVG code for a rectangle that aligns with Potrace formatting
                svg_data = (f'''<?xml version="1.0" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN"
     "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
    <svg version="1.0" xmlns="http://www.w3.org/2000/svg"
     width="100%" height="100%"
     viewBox="0 0 {width}.000000 {height}.000000"
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
                # Open the file, write the data and close it
                with open(svg_filename, "w", encoding="utf-8") as f:
                    f.write(svg_data)

                print("Made bottom layer a rectangle.\n")

            else:   # If it's not the bottom layer
                print("Converting PNG to SVG using Potrace...")

                print("Setting up temporary files...")
                temp_pbm = os.path.join(temp_dir, f"mask_{i}.pbm")  # Create an empty bitmap file in the temporary folder
                Image.fromarray(layer).convert("1").save(temp_pbm)  # Turn the array into a PBM file and saves it
                print("Set up temporary files.\n")

                # Arguments for the Potrace process
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
                subprocess.run(arguments, check=True)   # Run Potrace with the given arguments
                print("Converted PNG to SVG using Potrace.\n")

                print("Adding colours...")
                RGB_colour = cv2.cvtColor(numpy.uint8([[colour_groups[i]]]), cv2.COLOR_Lab2RGB)[0][0]   # Convert the layer's colour to RGB
                HEX_colour = ('#%02x%02x%02x' % tuple(int(c) for c in RGB_colour))  # Convert the RGB colours into RGB Hex
                print(f"Layer colour:{HEX_colour}")
                with open(svg_filename, "r", encoding="utf-8") as svg_data: # Open the SVG file
                    svg_data = svg_data.read()  # Read its contents
                    svg_data = svg_data.replace('fill="#000000"', f'fill="{HEX_colour}"')   # Replace the colour value with the correct colour
                print("Added colours.\n")

                print(f"Vectorised layer {i + 1}.\n")

                print(f"Saving layer {i+1}...")
                # Write the data to the file
                with open(svg_filename, "w", encoding="utf-8") as f:
                    f.write(svg_data)
                print(f"Saved layer {i + 1}.\n")

    print("Vectorised layers.\n")


# Combine and saves the SVG layers
def combine_svgs(image_path, output_path, colour_depth):
    title("Combining SVGs")

    print("Formatting layers...\n")
    combined_svg_code = ""  # Initialise an empty string for the combined code
    for i in range(colour_depth):   # Loop for each colour/layer
        subtitle(f"Layer {i+1}")

        # Open the file containing the code for that layer
        print("Opening file...")
        svg_filename = os.path.join(output_path, "SVG_layers_folder", f"{image_path}_layer_{i + 1}.svg")
        with open(svg_filename, "r", encoding="utf-8") as file:
            svg_code = file.read()
        print("Opened file.\n")

        print("Formatting code...")
        substring_index = svg_code.find("</metadata>") + 12     # Index of the start of the path code
        if i == 0:  # If it's the bottom layer
            # Take the root (beginning) of the SVG code and stores it to be appended later
            svg_root = svg_code[:substring_index]
        svg_code = svg_code[substring_index:-6]     # Stores the actual SVG code with all the paths.
        print("Formatted code.\n")

        print("Appending code...")
        combined_svg_code += svg_code   # Append this layer's paths
        print("Appended code.\n")

    print("Formatted layers.\n")

    combined_svg_code = svg_root + combined_svg_code + "</svg>"     # Combine the paths with the root and close the section

    combined_svg_code = combined_svg_code.replace("<<", "<")    # Fixe discrepancies

    print("Saving layers...")
    # Write the code to the file
    svg_filename = os.path.join(output_path, f"{image_path}_combined.svg")
    with open(svg_filename, "w", encoding="utf-8") as file:
        file.write(combined_svg_code)
    print("Saved layers.\n")


# Call the appropriate functions with the appropriate arguments
def main():
    image_path, colour_depth, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width = get_inputs()
    start = time.perf_counter()     # Start the timer once the inputs have been given
    image, output_path = load_image(image_path)
    image_path = image_path[:image_path.rfind(".")]     # Change the image path so to avoid .ext.ext files
    image, colour_groups, pixel_labels, colour_groups_order, height, width = quantise_image(image, colour_depth, image_path, output_path)
    layers = build_binary_layers(pixel_labels, image_path, output_path, colour_depth)
    bridged_layers, total_contours = clean_contours(layers, image_path, output_path, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width)
    convert_layers_to_svg(bridged_layers, image_path, colour_groups, output_path, height, width, colour_depth, max_bridge_contour_area, min_contour_area, max_contour_distance, bridge_width)
    combine_svgs(image_path, output_path, colour_depth)

    end = time.perf_counter()   # Stop the timer and display the results
    print(f"Time: {end - start:.6f} seconds.\n")
    print(f"Total contours: {total_contours}.\n")
    title("Complete")


main()
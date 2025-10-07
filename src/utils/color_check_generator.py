


def generate_simple_color_check(
    width: int = 1748, height: int = 2480, filename: str = "simple_color_check.png"
):
    """
    Generates a simple color check image with primary and secondary colors.

    The image consists of a grid of color patches, each labeled with its color name.
    Default dimensions are set for A5 paper at 300 DPI.

    Args:
        width (int): The width of the image in pixels. Defaults to 1748.
        height (int): The height of the image in pixels. Defaults to 2480.
        filename (str): The name of the file to save the image as. Defaults to "simple_color_check.png".
    """
    # Define the primary and secondary colors, plus white
    colors = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "White": (255, 255, 255),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "Yellow": (255, 255, 0),
    }

    # Create a new black image to serve as the background
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)

    # Set up layout parameters for the grid
    cols = 3
    rows = 3
    margin = 50  # Outer margin
    gutter = 20  # Space between patches

    # Calculate the size of each color patch based on image dimensions and layout
    patch_width = (width - 2 * margin - (cols - 1) * gutter) / cols
    patch_height = (height - 2 * margin - (rows - 1) * gutter) / rows

    # Try to load a common font, otherwise fall back to the default
    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except IOError:
        font = ImageFont.load_default()

    # Iterate over the colors and draw them on the image
    color_items = list(colors.items())
    for i, (name, color) in enumerate(color_items):
        col = i % cols
        row = i // cols

        # Calculate the top-left corner of the patch
        x0 = margin + col * (patch_width + gutter)
        y0 = margin + row * (patch_height + gutter)

        # Calculate the bottom-right corner of the patch
        x1 = x0 + patch_width
        y1 = y0 + patch_height

        # Draw the color patch
        draw.rectangle([x0, y0, x1, y1], fill=color)

        # Draw the text label for the color
        text_pos = (x0 + 20, y1 - 60)
        draw.text(text_pos, name, font=font, fill="black")

    # Save the final image to a file
    img.save(filename)
    print(f"Simple color check image saved as {filename}")


def generate_16_color_check(
    width: int = 1748, height: int = 2480, filename: str = "16_color_check.png"
):
    """
    Generates a color check image with 16 distinct colors arranged in a 4x4 grid.

    Each color patch is labeled with its name, and text color is adjusted for contrast.
    Default dimensions are set for A5 paper at 300 DPI.

    Args:
        width (int): The width of the image in pixels. Defaults to 1748.
        height (int): The height of the image in pixels. Defaults to 2480.
        filename (str): The name of the file to save the image as. Defaults to "16_color_check.png".
    """
    # Define a palette of 16 colors
    colors = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "White": (255, 255, 255),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "Yellow": (255, 255, 0),
        "Orange": (255, 165, 0),
        "Purple": (128, 0, 128),
        "Brown": (165, 42, 42),
        "Pink": (255, 192, 203),
        "Teal": (0, 128, 128),
        "Lime": (191, 255, 0),
        "Navy": (0, 0, 128),
        "Olive": (128, 128, 0),
        "Maroon": (128, 0, 0),
    }

    # Create a new black image
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)

    # Set up layout parameters for a 4x4 grid
    cols = 4
    rows = 4
    margin = 50
    gutter = 20

    # Calculate the size of each color patch
    patch_width = (width - 2 * margin - (cols - 1) * gutter) / cols
    patch_height = (height - 2 * margin - (rows - 1) * gutter) / rows

    # Try to load a common font, otherwise fall back to the default
    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except IOError:
        font = ImageFont.load_default()

    # Iterate over the colors and draw them on the image
    color_items = list(colors.items())
    for i, (name, color) in enumerate(color_items):
        col = i % cols
        row = i // cols

        # Calculate patch coordinates
        x0 = margin + col * (patch_width + gutter)
        y0 = margin + row * (patch_height + gutter)
        x1 = x0 + patch_width
        y1 = y0 + patch_height

        # Draw the color patch
        draw.rectangle([x0, y0, x1, y1], fill=color)

        # Determine text color based on background brightness for better contrast
        brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
        text_color = "black" if brightness > 128 else "white"

        # Draw the text label for the color
        text_pos = (x0 + 20, y1 - 60)
        draw.text(text_pos, name, font=font, fill=text_color)

    # Save the image
    img.save(filename)
    print(f"16-color check image saved as {filename}")


def generate_comprehensive_color_check(
    width: int = 1748, height: int = 2480, filename: str = "comprehensive_color_check.png"
):
    """
    Generates a comprehensive color check image with multiple sections:
    - Primary and secondary colors
    - A full grayscale ramp
    - A palette of common named colors
    - Three smooth color gradients

    Default dimensions are set for A5 paper at 300 DPI.

    Args:
        width (int): The width of the image in pixels. Defaults to 1748.
        height (int): The height of the image in pixels. Defaults to 2480.
        filename (str): The name of the file to save the image as. Defaults to "comprehensive_color_check.png".
    """
    # Create a new black image
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)

    # Load fonts for titles and labels
    try:
        font = ImageFont.truetype("arial.ttf", size=30)
        title_font = ImageFont.truetype("arial.ttf", size=50)
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    margin = 50
    y_cursor = margin

    # --- Section 1: Primary and Secondary Colors ---
    draw.text(
        (margin, y_cursor), "Primary & Secondary Colors", font=title_font, fill="white"
    )
    y_cursor += 70

    primary_colors = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "Yellow": (255, 255, 0),
    }
    patch_width = (width - 2 * margin) / len(primary_colors)
    patch_height = 200
    for i, (name, color) in enumerate(primary_colors.items()):
        x0 = margin + i * patch_width
        y0 = y_cursor
        x1 = x0 + patch_width
        y1 = y0 + patch_height
        draw.rectangle([x0, y0, x1, y1], fill=color)
        draw.text((x0 + 10, y1 - 40), name, font=font, fill="black")
    y_cursor += patch_height + 50

    # --- Section 2: Grayscale Ramp ---
    draw.text((margin, y_cursor), "Grayscale Ramp", font=title_font, fill="white")
    y_cursor += 70

    gray_height = 200
    ramp_width = width - 2 * margin
    # Create the grayscale ramp using numpy for efficiency
    gray_ramp = np.tile(
        np.linspace(0, 255, ramp_width, dtype=np.uint8), (gray_height, 1)
    )
    gray_ramp_img = Image.fromarray(gray_ramp, "L").convert("RGB")
    img.paste(gray_ramp_img, (margin, y_cursor))
    y_cursor += gray_height + 50

    # --- Section 3: Common Colors ---
    draw.text((margin, y_cursor), "Common Colors", font=title_font, fill="white")
    y_cursor += 70
    common_colors = {
        "Orange": (255, 165, 0),
        "Purple": (128, 0, 128),
        "Brown": (165, 42, 42),
        "Pink": (255, 192, 203),
        "Teal": (0, 128, 128),
        "Lime": (191, 255, 0),
        "Navy": (0, 0, 128),
        "Olive": (128, 128, 0),
        "Maroon": (128, 0, 0),
    }
    patch_width = (width - 2 * margin) / len(common_colors)
    patch_height = 200
    for i, (name, color) in enumerate(common_colors.items()):
        brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
        text_color = "black" if brightness > 128 else "white"
        x0 = margin + i * patch_width
        y0 = y_cursor
        x1 = x0 + patch_width
        y1 = y0 + patch_height
        draw.rectangle([x0, y0, x1, y1], fill=color)
        draw.text((x0 + 10, y1 - 40), name, font=font, fill=text_color)
    y_cursor += patch_height + 50

    # --- Section 4: Color Gradients ---
    draw.text((margin, y_cursor), "Color Gradients", font=title_font, fill="white")
    y_cursor += 70

    gradient_height = 200
    gradient_width = width - 2 * margin

    # Function to create and paste a gradient
    def create_and_paste_gradient(r_ch, g_ch, b_ch, y_pos):
        # Stack color channels and reshape for an image
        gradient_arr = np.dstack((r_ch, g_ch, b_ch)).reshape(1, gradient_width, 3)
        # Repeat the 1px high gradient to the desired height
        gradient_arr = np.repeat(gradient_arr, gradient_height, axis=0)
        gradient_img = Image.fromarray(gradient_arr, "RGB")
        img.paste(gradient_img, (margin, y_pos))
        return y_pos + gradient_height + 20

    # Gradient 1: Red to Green
    r = np.linspace(255, 0, gradient_width, dtype=np.uint8)
    g = np.linspace(0, 255, gradient_width, dtype=np.uint8)
    b = np.zeros(gradient_width, dtype=np.uint8)
    y_cursor = create_and_paste_gradient(r, g, b, y_cursor)

    # Gradient 2: Green to Blue
    r = np.zeros(gradient_width, dtype=np.uint8)
    g = np.linspace(255, 0, gradient_width, dtype=np.uint8)
    b = np.linspace(0, 255, gradient_width, dtype=np.uint8)
    y_cursor = create_and_paste_gradient(r, g, b, y_cursor)

    # Gradient 3: Blue to Red
    r = np.linspace(0, 255, gradient_width, dtype=np.uint8)
    g = np.zeros(gradient_width, dtype=np.uint8)
    b = np.linspace(255, 0, gradient_width, dtype=np.uint8)
    y_cursor = create_and_paste_gradient(r, g, b, y_cursor)

    # Save the final image
    img.save(filename)
    print(f"Comprehensive color check image saved as {filename}")


# --- Main execution block ---
if __name__ == "__main__":
    # This block runs when the script is executed directly.
    # You can choose which color chart to generate.

    # A5 paper dimensions in pixels at 300 DPI are a good default
    # Width: (148mm / 25.4mm/in) * 300dpi = 1748 px
    # Height: (210mm / 25.4mm/in) * 300dpi = 2480 px
    A5_WIDTH, A5_HEIGHT = 1748, 2480

    print("Generating color check images...")

    # --- Uncomment the function call for the image you want to create ---

    # 1. Generate the simple version with default A5 size
    # generate_simple_color_check(width=A5_WIDTH, height=A5_HEIGHT)

    # 2. Generate the 16-color version with default A5 size
    # generate_16_color_check(width=A5_WIDTH, height=A5_HEIGHT)

    # 3. Generate the comprehensive version with default A5 size
    generate_comprehensive_color_check(width=A5_WIDTH, height=A5_HEIGHT)

    # Example of generating a smaller, custom-sized image
    # print("\nGenerating a smaller custom image...")
    # generate_comprehensive_color_check(width=1200, height=800, filename="comprehensive_check_small.png")

    print("\nAll tasks complete.")

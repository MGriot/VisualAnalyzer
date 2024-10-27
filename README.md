# Photo Color Analyzer - Computer Vision Module

This module contains tools for analyzing images, particularly focusing on color analysis and clustering.

## Structure

The module is organized into the following subfolders:

- `imgAnalyzer`: Contains classes for various image analysis tasks, including:
    - `COCO`: Interacts with the COCO dataset.
    - `ImageAnalyzer`: Compares and analyzes images.
    - `ImageCluster`: Clusters images based on color.
    - `ImageProcessor`: Performs various image processing tasks.
- `img`: Stores images used for analysis or testing.
- `output`: Holds the results of the analysis.

## Usage

1. Install the required libraries listed in `requirements.txt` using:
   ```bash
   pip install -r requirements.txt
   ```
2. Use the provided classes and functions to analyze images. For example, to cluster an image based on color:
   ```python
   from imgAnalyzer import ImageCluster

   clusterer = ImageCluster("path/to/your/image.jpg")
   clusterer.cluster(n_clusters=5)
   clusterer.save_plots()
   ```

## Examples

See the individual files within the `imgAnalyzer` subfolder for example usage of each class.

## Contributing

Feel free to contribute to this module by adding new features, improving existing code, or providing more comprehensive documentation.

## License

This module is licensed under the [MIT License](LICENSE).

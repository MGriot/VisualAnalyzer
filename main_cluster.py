from VisualAnalyzer.ImageAnalyzer import ImageAnalyzer
from VisualAnalyzer.ImageCluster import ImageCluster
from VisualAnalyzer.ImageProcessor import ImageProcessor
import numpy as np

path = r"C:\Users\Admin\Documents\Coding\VisualAnalyzer\.old\img\test2.png"


# Uso della classe ImageProcessor
processor = ImageProcessor(path)
# processor.equalize()
processor.blur_filter("GaussianBlur", radius=5)
processor.show_image()
processor.resize(size=400, maintain_aspect_ratio=True)
processor.show_image()
blurred_img = processor.img

# Uso della classe ImageAnalyzer
analyzer = ImageAnalyzer(
    np.array(blurred_img),
    np.array(blurred_img),
)
analyzer.analyze()

# Uso della classe ImageCluster
c = ImageCluster(blurred_img)
c.remove_transparent()
c.cluster(n_clusters=3)
c.plot_clustered_image()
c.extract_cluster_info()
c.plot_clustered_image_high_contrast()
c.plot_images()
c.plot_cluster_pie()

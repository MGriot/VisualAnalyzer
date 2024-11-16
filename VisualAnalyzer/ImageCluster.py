# Librerie per l'elaborazione delle immagini
from PIL import Image
import numpy as np

# Librerie per il machine learning
from sklearn.cluster import KMeans
from scipy.spatial import distance

# Librerie per la visualizzazione dei dati
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Altre librerie
from collections import Counter
import os


# aggiungi una caratteristica tipo nome file (Senza estensione) in modo da usarlo per salvare tutti i grafici
class ImageCluster:
    """
    This class provides methods to perform color clustering on an image and 
    analyze the color distribution. It uses KMeans clustering to group similar 
    colors together and provides various visualization tools to understand 
    the results.

    Attributes:
        image_input (str or PIL.Image.Image): The input image, either a file path or a PIL.Image object.
        n_clusters (int): The number of clusters to form.
        initial_clusters (np.ndarray): Initial cluster centers, if provided.
        img_array (np.ndarray): The image data as a NumPy array.
        data (np.ndarray): Reshaped image data for clustering.
        removeTransparent (bool): Flag indicating if transparent pixels have been removed.
        labels_full (np.ndarray): Cluster labels for all pixels, including transparent ones.
        mask (np.ndarray): Boolean mask indicating non-transparent pixels.
        clustered_img (np.ndarray): The clustered image, where each pixel is replaced with its cluster's color.
        cluster_infos (dict): Information about each cluster, including color, pixel count, and percentage.

    Methods:
        remove_transparent(alpha_threshold=250): Removes transparent pixels from the image.
        filter_alpha(): Returns a boolean mask indicating non-transparent pixels.
        cluster(n_clusters=None, initial_clusters=None, merge_similar=False, threshold=10): Performs color clustering.
        create_clustered_image(): Creates an image where each pixel is replaced with its cluster's color.
        create_clustered_image_with_ids(): Creates an image where each pixel is replaced with its cluster's ID.
        extract_cluster_info(): Extracts information about the clusters.
        calculate_brightness(color): Calculates the brightness of a color.
        plot_original_image(ax=None, max_size=(1024, 1024)): Displays the original image.
        plot_clustered_image(ax=None, max_size=(1024, 1024)): Displays the clustered image.
        plot_clustered_image_high_contrast(style='jet', show_percentage=True, dpi=100, ax=None): Displays the clustered image with high contrast.
        plot_cluster_pie(ax=None, dpi=100): Displays a pie chart of cluster distribution.
        plot_cluster_bar(ax=None, dpi=100): Displays a bar chart of cluster distribution.
        plot_cumulative_barchart(ax=None, dpi=100): Displays a cumulative bar chart of cluster distribution.
        plot_images(max_size=(1024, 1024)): Displays the original, clustered, and high contrast clustered images.
        plot_image_with_grid(grid_size=50, color='white', max_size=(1024, 1024), dpi=100): Displays the original image with a grid overlaid.
        save_plots(): Saves all generated plots to a directory.
        get_dominant_color(): Returns the dominant color of the image.
    """

    def __init__(self, image_input):
        """
        Initializes the ImageCluster object.
        If image_input is a string, it is treated as a file path and the image is loaded from that path.
        If image_input is an instance of PIL.Image, it is used directly.
        """
        if isinstance(image_input, str):
            self.image_path = image_input
            self.filename = os.path.splitext(os.path.basename(self.image_path))[0]
            self.img = Image.open(self.image_path).convert("RGBA")
        elif isinstance(image_input, Image.Image):
            self.img = image_input.convert("RGBA")
            self.filename = "image"
        else:
            raise TypeError(
                "image_input deve essere un percorso dell'immagine o un'istanza di PIL.Image"
            )
        self.n_clusters = None
        self.initial_clusters = None
        self.img_array = np.array(self.img)
        self.data = self.img_array.reshape(-1, 4)
        self.data = self.data.astype(float)

        self.removeTransparent = False
        self.labels_full = None
        self.mask = None
        self.clustered_img = None
        self.cluster_infos = None

    def remove_transparent(self, alpha_threshold=250):
        """
        Removes transparent pixels from the image.
        A pixel is considered transparent if its alpha value is less than alpha_threshold.
        """
        transparent_pixels = self.data[:, 3] <= alpha_threshold
        self.data[transparent_pixels] = np.nan
        self.removeTransparent = True

    def filter_alpha(self):
        """
        Returns a boolean mask indicating which pixels in the image are not transparent.
        """
        return ~np.isnan(self.data[:, 3])

    def cluster(
        self, n_clusters=None, initial_clusters=None, merge_similar=False, threshold=10
    ):
        """
        Performs color clustering on the image.
        If initial_clusters is provided, it is used as initialization for the KMeans algorithm.
        Otherwise, if n_clusters is provided, it is used to determine the number of clusters.
        If merge_similar is True, clusters with similar colors are merged.
        The threshold for determining whether two colors are similar is given by threshold.
        """
        self.initial_clusters = initial_clusters
        if initial_clusters is not None:
            self.n_clusters = self.initial_clusters.shape[0]
        else:
            if n_clusters is not None:
                self.n_clusters = n_clusters
            else:
                print("Error, choice cluster number n_clusters")

        mask = self.filter_alpha()
        self.mask = mask
        data_no_nan = self.data[mask]
        if self.initial_clusters is not None:
            kmeans = KMeans(
                n_clusters=self.n_clusters, init=self.initial_clusters, n_init=10
            )
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.labels = kmeans.fit_predict(data_no_nan[:, :3])  # Ignora la colonna alpha
        self.center_colors = kmeans.cluster_centers_
        self.labels_full = np.full(mask.shape[0], -1)
        self.labels_full[mask] = self.labels

        if merge_similar:
            while True:
                # Calcola la distanza euclidea tra i colori dei centri dei cluster
                distances = distance.cdist(
                    self.center_colors, self.center_colors, "euclidean"
                )
                # Trova la distanza minima che non sia sulla diagonale
                min_distance = np.min(
                    distances + np.eye(distances.shape[0]) * distances.max()
                )
                if min_distance >= threshold:
                    break
                else:
                    self.n_clusters -= 1
                    kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
                    self.labels = kmeans.fit_predict(
                        data_no_nan[:, :3]
                    )  # Ignora la colonna alpha
                    self.center_colors = kmeans.cluster_centers_
                    self.labels_full = np.full(mask.shape[0], -1)
                    self.labels_full[mask] = self.labels

    def create_clustered_image(self):
        """
        Creates an image where each pixel is replaced with the color of its cluster.
        """
        self.clustered_img = np.zeros_like(self.img_array)
        for i in range(self.img_array.shape[0]):
            for j in range(self.img_array.shape[1]):
                if self.labels_full[i * self.img_array.shape[1] + j] != -1:
                    self.clustered_img[i, j, :3] = self.center_colors[
                        self.labels_full[i * self.img_array.shape[1] + j]
                    ]
                    self.clustered_img[i, j, 3] = self.data[
                        i * self.img_array.shape[1] + j, 3
                    ]  # Mantieni il valore alfa originale
                else:
                    self.clustered_img[i, j] = [
                        255,
                        255,
                        255,
                        0,
                    ]  # white or transparent

    def create_clustered_image_with_ids(self):
        """
        Creates an image where each pixel is replaced with the ID of its cluster.
        """
        # Inizializza un array bidimensionale con la stessa forma di self.img_array
        self.clustered_img_with_ids = np.zeros(
            (self.img_array.shape[0], self.img_array.shape[1])
        )
        for i in range(self.img_array.shape[0]):
            for j in range(self.img_array.shape[1]):
                if self.labels_full[i * self.img_array.shape[1] + j] != -1:
                    # Rimpiazza il colore con l'ID del cluster
                    self.clustered_img_with_ids[i, j] = self.labels_full[
                        i * self.img_array.shape[1] + j
                    ]
                else:
                    self.clustered_img_with_ids[i, j] = self.n_clusters + 1

    def extract_cluster_info(self):
        """
        Extracts information about the clusters, such as the color of the centroid, the number of pixels, and the percentage of total pixels.
        """
        counter = Counter(self.labels)
        clusters_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        cluster_info = {}
        total_pixels = sum(counter.values())
        for i, (cluster, count) in enumerate(clusters_sorted):
            cluster_info[i] = {
                "color": self.center_colors[cluster],
                "pixel_count": count,
                "total_pixel_percentage": (count / total_pixels) * 100,
                "original_position": cluster,
            }
        cluster_info = dict(
            sorted(
                cluster_info.items(),
                key=lambda item: item[1]["pixel_count"],
                reverse=True,
            )
        )
        self.cluster_infos = cluster_info
        self.total_pixels = total_pixels

    def calculate_brightness(self, color):
        """
        Calculates the brightness of a color.
        Brightness is defined as the average of the RGB values.
        """
        # Calcola la luminosità come la media dei valori RGB
        return sum(color) / (3 * 255)

    def plot_original_image(self, ax=None, max_size=(1024, 1024)):
        """
        Displays the original image.
        If ax is provided, the image is displayed on that subplot.
        Otherwise, a new subplot is created.
        The image is resized to max_size to avoid using too much memory.
        """
        # Riduci la risoluzione dell'immagine
        img = self.img.copy()
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        if ax is None:
            ax = plt.gca()
        ax.imshow(np.array(img))
        ax.set_title("Original Image")
        ax.axis("off")

    def plot_clustered_image(self, ax=None, max_size=(1024, 1024)):
        """
        Displays the clustered image.
        If ax is provided, the image is displayed on that subplot.
        Otherwise, a new subplot is created.
        The image is resized to max_size to avoid using too much memory.
        """
        # Pre-calcola l'immagine raggruppata
        if self.clustered_img is None:
            self.create_clustered_image()

        # Riduci la risoluzione dell'immagine raggruppata
        img = Image.fromarray(self.clustered_img).convert("RGBA")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        if ax is None:
            ax = plt.gca()
        ax.imshow(np.array(img))
        ax.set_title("Clustered Image ({} clusters)".format(self.n_clusters))
        ax.axis("off")

    def plot_clustered_image_high_contrast(
        self, style="jet", show_percentage=True, dpi=100, ax=None
    ):
        """
        Displays the clustered image with high contrast between the cluster colors.
        The style parameter determines the colormap used.
        If show_percentage is True, the percentage of pixels in each cluster is displayed in the legend.
        """
        # Prima assicurati di aver chiamato la funzione create_clustered_image_with_ids
        self.create_clustered_image_with_ids()

        # Crea una nuova figura con i DPI specificati
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)

        # Visualizza l'immagine
        im = ax.imshow(self.clustered_img_with_ids, cmap=style)

        # Crea una legenda con un rettangolo colorato per ogni etichetta di cluster
        colors = [
            im.cmap(im.norm(self.cluster_infos[i]["original_position"]))
            for i in range(self.n_clusters)
        ]
        if show_percentage:
            labels = [
                f"Cluster {self.cluster_infos[i]['original_position']} ({self.cluster_infos[i]['total_pixel_percentage']:.2f}%)"
                for i in range(self.n_clusters)
                if i in self.cluster_infos
            ]
        else:
            labels = [
                f"Cluster {self.cluster_infos[i]['original_position']}"
                for i in range(self.n_clusters)
            ]
        patches = [
            mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))
        ]
        plt.legend(
            handles=patches,
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0.0,
            title="Legend",
        )

        ax.set_title(
            "Clustered Image with High Contrast ({} clusters)".format(self.n_clusters)
        )
        ax.axis("off")

        # Mostra la figura
        if ax is None:
            plt.show()

    def plot_cluster_pie(self, ax=None, dpi=100):
        """
        Displays a pie chart showing the distribution of pixels among the clusters.
        If ax is provided, the chart is displayed on that subplot.
        Otherwise, a new subplot is created.
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)
        labels = [
            f"Cluster {self.cluster_infos[i]['original_position']}"
            for i in range(self.n_clusters)
            if i in self.cluster_infos
        ]
        sizes = [
            self.cluster_infos[i]["pixel_count"]
            for i in range(self.n_clusters)
            if i in self.cluster_infos
        ]
        colors = [
            self.cluster_infos[i]["color"] / 255
            for i in range(self.n_clusters)
            if i in self.cluster_infos
        ]
        wedges, text_labels, text_percentages = ax.pie(
            sizes, labels=labels, colors=colors, startangle=90, autopct="%1.1f%%"
        )
        for i in range(len(wedges)):
            color = "white" if self.calculate_brightness(colors[i]) < 0.5 else "black"
            text_labels[i].set_color(color)
            text_percentages[i].set_color(color)
        ax.legend(
            wedges,
            labels,
            title="Clusters",
            loc="best",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
        )
        ax.axis("equal")
        ax.set_title("PieChart ({} clusters)".format(self.n_clusters))
        plt.show()

    def plot_cluster_bar(self, ax=None, dpi=100):
        """
        Displays a bar chart showing the distribution of pixels among the clusters.
        If ax is provided, the chart is displayed on that subplot.
        Otherwise, a new subplot is created.
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)
        labels = [f"Cluster {i}" for i in self.cluster_infos.keys()]
        percentages = [
            info["total_pixel_percentage"] for info in self.cluster_infos.values()
        ]
        pixel_counts = [info["pixel_count"] for info in self.cluster_infos.values()]
        colors = [info["color"] / 255 for info in self.cluster_infos.values()]
        bars = ax.bar(labels, percentages, color=colors)
        for bar, pixel_count in zip(bars, pixel_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(pixel_count),
                ha="center",
                va="bottom",
            )

        ax.set_xlabel("Cluster")
        ax.set_ylabel("Percentage")

    def plot_cumulative_barchart(self, ax=None, dpi=100):
        """
        Displays a cumulative bar chart showing the distribution of pixels among the clusters.
        If ax is provided, the chart is displayed on that subplot.
        Otherwise, a new subplot is created.
        """
        if ax is None:
            fig, ax = plt.subplots(dpi=dpi)
        bottom = 0
        for i, info in self.cluster_infos.items():
            color = info["color"] / 255
            percentage = info["total_pixel_percentage"]
            pixel_count = info["pixel_count"]
            ax.bar("Cluster", height=percentage, color=color, bottom=bottom)
            brightness = self.calculate_brightness(color)
            text_color = "white" if brightness < 0.75 else "black"
            ax.text(
                "Cluster",
                bottom + percentage / 2,
                str(pixel_count),
                ha="center",
                va="center",
                color=text_color,
            )
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            bottom += percentage
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.axes.xaxis.set_visible(False)

    def plot_images(self, max_size=(1024, 1024)):
        """
        Displays the original image, the clustered image, and the high contrast clustered image side by side.
        """
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        self.plot_original_image(ax=axs[0], max_size=max_size)
        self.plot_clustered_image(ax=axs[1], max_size=max_size)
        self.plot_clustered_image_high_contrast(ax=axs[2])
        plt.show()

    def plot_image_with_grid(
        self, grid_size=50, color="white", max_size=(1024, 1024), dpi=100
    ):
        """
        Displays the original image with a grid overlaid.
        The grid size is determined by grid_size.
        The grid color is determined by color.
        The image is resized to max_size to avoid using too much memory.
        """
        fig, ax = plt.subplots(dpi=dpi)

        # Riduci la risoluzione dell'immagine originale
        img = self.img.copy()
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Mostra l'immagine originale
        ax.imshow(np.array(img))

        # Aggiungi la griglia
        ax.set_xticks(np.arange(-0.5, img.size[0], grid_size), minor=True)
        ax.set_yticks(np.arange(-0.5, img.size[1], grid_size), minor=True)
        ax.grid(which="minor", color=color, linestyle="-", linewidth=2)

        # Imposta il titolo e nasconde gli assi
        ax.set_title("Original Image with Grid")
        ax.axis("on")

        plt.show()

    def save_plots(self):
        """
        Saves all the plots in a directory named "output/{self.filename}".
        If the directory does not exist, it is created.
        """
        # Crea la directory se non esiste
        if not os.path.exists(f"output/{self.filename}"):
            os.makedirs(f"output/{self.filename}")
        self.plot_original_image()
        plt.savefig(f"output/{self.filename}/{self.filename}.png")
        self.plot_clustered_image()
        plt.savefig(f"output/{self.filename}/{self.filename}_cluster_image.png")
        self.plot_cluster_pie()
        plt.savefig(f"output/{self.filename}/{self.filename}_piechart.png")
        self.plot_clustered_image_high_contrast()
        plt.savefig(f"output/{self.filename}/{self.filename}_high_contrast.png")

    def get_dominant_color(self):
        """
        Returns the dominant color of the image, which is the color of the cluster with the most pixels.
        This method should be called after the cluster() method has been called.

        Returns:
            np.ndarray: The RGB color of the dominant cluster.
        """
        if self.cluster_infos is None:
            raise ValueError("The cluster() method must be called before get_dominant_color().")
        return self.cluster_infos[0]['color']

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Define a professional color palette
COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def radarChart(
    df,
    *,
    id_column,
    title=None,
    max_values=None,
    padding=1.25,
    frame="circle",
    fill_area=True,
):
    """
    Creates a customizable radar chart.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        id_column (str): Name of the column containing the IDs for each data series.
        title (str, optional): Title of the chart. Defaults to None.
        max_values (dict, optional): Dictionary specifying maximum values for each category. Defaults to None.
        padding (float, optional): Padding factor for the radial axis limits. Defaults to 1.25.
        frame (str, optional): Shape of the frame surrounding the chart. Can be 'circle' or 'polygon'. Defaults to 'circle'.
        fill_area (bool, optional): Whether to fill the area inside the lines. Defaults to True.
    """
    categories = df._get_numeric_data().columns.tolist()
    data = df[categories].to_dict(orient="list")
    ids = df[id_column].tolist()
    if max_values is None:
        max_values = {key: padding * max(value) for key, value in data.items()}

    normalized_data = {
        key: np.array(value) / max_values[key] for key, value in data.items()
    }
    num_vars = len(data.keys())
    tiks = list(data.keys())
    tiks += tiks[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist() + [0]

    # Set a professional font
    plt.rcParams["font.family"] = "Arial"

    if frame == "circle":
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for i, model_name in enumerate(ids):
            values = [normalized_data[key][i] for key in data.keys()]
            actual_values = [data[key][i] for key in data.keys()]
            values += values[:1]  # Close the plot for a better look

            # Use different line styles and colors from the palette
            ax.plot(
                angles,
                values,
                label=model_name,
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                linestyle="-" if i % 2 == 0 else "--",
            )
            if fill_area:
                ax.fill(
                    angles,
                    values,
                    alpha=0.2,
                    color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                )

            for _x, _y, t in zip(angles, values, actual_values):
                t = f"{t:.2f}" if isinstance(t, float) else str(t)
                ax.text(_x, _y, t, size="small")

        if fill_area:
            ax.fill(angles, np.ones(num_vars + 1), alpha=0.05, color="gray")
        ax.set_yticklabels([])
        ax.set_xticks(angles)
        ax.set_xticklabels(tiks)

        # Customize gridlines
        ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    elif frame == "polygon":
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.set_axis_off()

        for i, model_name in enumerate(ids):
            values = [normalized_data[key][i] for key in data.keys()]
            actual_values = [data[key][i] for key in data.keys()]
            values += values[:1]  # Close the plot for a better look

            # Convert polar coordinates to cartesian
            x = [v * np.cos(a) for v, a in zip(values, angles)]
            y = [v * np.sin(a) for v, a in zip(values, angles)]

            # Use different line styles and colors from the palette
            ax.plot(
                x,
                y,
                label=model_name,
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                linestyle="-" if i % 2 == 0 else "--",
            )
            if fill_area:
                ax.fill(x, y, alpha=0.2, color=COLOR_PALETTE[i % len(COLOR_PALETTE)])

            for _x, _y, t in zip(x, y, actual_values):
                t = f"{t:.2f}" if isinstance(t, float) else str(t)
                ax.text(_x, _y, t, size="small")

        # Draw the polygon frame
        frame_x = [np.cos(a) for a in angles]
        frame_y = [np.sin(a) for a in angles]
        ax.plot(frame_x, frame_y, color="gray", linestyle="-", linewidth=0.5)

        # Draw category lines
        for a in angles[:-1]:
            ax.plot(
                [0, np.cos(a)],
                [0, np.sin(a)],
                color="gray",
                linestyle="-",
                linewidth=0.5,
                alpha=0.5,
            )

        # Set tick labels
        for a, t in zip(angles[:-1], tiks[:-1]):
            ax.text(
                1.1 * np.cos(a),
                1.1 * np.sin(a),
                t,
                size="medium",
                ha="center",
                va="center",
            )

    else:
        raise ValueError("Invalid frame shape. Choose either 'circle' or 'polygon'.")

    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    if title is not None:
        plt.title(title, fontsize=16, fontweight="bold")
    plt.show()


radar = radarChart

radarChart(
    pd.DataFrame(
        {
            "x": [*"abcde"],
            "c1": [10, 11, 12, 13, 14],
            "c2": [0.1, 0.3, 0.4, 0.1, 0.9],
            "c3": [1e5, 2e5, 3.5e5, 8e4, 5e4],
            "c4": [9, 12, 5, 2, 0.2],
            "test": [1, 1, 1, 1, 5],
        }
    ),
    id_column="x",
    title="Sample Spider",
    padding=1.1,
    frame="circle",
    fill_area=True,
)

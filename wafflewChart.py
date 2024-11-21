# https://python-graph-gallery.com/web-waffle-chart-as-share/
#
# # Libraries
import matplotlib.pyplot as plt
import pandas as pd
from pywaffle import Waffle
from highlight_text import fig_text, ax_text
from pyfonts import load_font, load_exact_font

def create_waffle_chart(
    df, background_color="#222725", pink="#f72585", dark_pink="#7a0325"
):
    """Creates a waffle chart for each row in the provided DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing data for the waffle charts.
            Each row should have a column named 'percent' representing the share to
            visualize.
        background_color (str, optional): The background color of the figure. Defaults to "#222725".
        pink (str, optional): The color for the positive share (represented by pink squares). Defaults to "#f72585".
        dark_pink (str, optional): The color for the negative share (represented by dark pink squares). Defaults to "#7a0325".

    Returns:
        None
    """

    num_rows = len(df)
    fig, axs = plt.subplots(nrows=num_rows, ncols=1, figsize=(8, 8 * num_rows))
    fig.set_facecolor(background_color)

    for i, row in df.iterrows():
        ax = axs[i]  # Select the appropriate axis for this row
        share = row["percent"]
        values = [share, 100 - share]  # Ensure values sum to 100

        Waffle.make_waffle(
            ax=ax,
            rows=4,
            columns=25,
            values=values,
            colors=[pink, dark_pink],
        )

        # Optional customizations (uncomment if desired)
        # ax.set_title(row['some_column_for_title'])  # Set title based on a column
        # ax.set_xlabel('Share')  # Add x-axis label
        # ax.set_ylabel('Remaining')  # Add y-axis label

    plt.tight_layout()  # Adjust spacing for multiple subplots
    plt.show()


# Sample DataFrame (replace with your actual data)
data = {
    "continent": ["Africa", "Asia", "Europe", "North America", "South America"],
    "percent": [20, 35, 15, 22, 8],
}
df = pd.DataFrame(data)

create_waffle_chart(df)

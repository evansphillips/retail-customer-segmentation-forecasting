import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def plot_donut_chart(
    category_counts: pd.Series,
    title: str,
    legend: str,
    threshold: Optional[int] = None
) -> None:
    """
    Plot a donut chart based on the distribution of categories in the given data.

    Parameters:
    - category_counts (pd.Series): A Series or DataFrame containing the category counts.
    - title (str): The title of the donut chart.
    - legend (str): The legend title.
    - threshold (Optional[int]): Threshold for grouping small categories into 'Other'. Default is None.

    Returns:
    None
    """
    # Identify categories below the threshold
    if threshold is not None:
        small_categories = category_counts[category_counts < threshold]

        # Group small categories into 'Other'
        category_counts['Other'] = small_categories.sum()
        category_counts = category_counts[category_counts >= threshold]

    # Plot the donut chart
    fig, ax = plt.subplots()

    # Draw the inner circle (donut hole)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Plot the donut chart
    wedges, texts, autotexts = ax.pie(
        category_counts,
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.3),
        pctdistance=0.85,
    )

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    # Add legend
    ax.legend(wedges, category_counts.index, title=legend.title(), loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # Add percentages inside the wedges
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(8)

    plt.title(title.title())
    plt.show()
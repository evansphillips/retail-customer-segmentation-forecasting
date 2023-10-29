from typing import Tuple, List
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt



def plot_monthly_revenue(retail_data: pd.DataFrame):
    """
    Plot average monthly revenue with highlighted months in a custom order.

    Parameters:
    - retail_data (pd.DataFrame): The input DataFrame containing 'month' and 'revenue' columns.

    Returns:
    None
    """
    # Convert 'month' to a categorical type with custom ordering
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    retail_data["month"] = pd.Categorical(
        retail_data["month"], categories=month_order, ordered=True
    )

    # Sort the DataFrame based on the custom ordering of 'month'
    avg_monthly_rev = retail_data.sort_values("month").groupby("month")["revenue"].mean()

    # Set the figure size to make it longer
    plt.figure(figsize=(10, 5))

    # Plot average revenue line
    plt.plot(
        avg_monthly_rev.index,
        avg_monthly_rev.values,
        marker="o",
        label="Average Monthly Revenue (£)",
    )

    # Months to highlight
    highlight_months = ["July", "September", "November", "December"]

    # Draw vertical lines for selected months
    for month in highlight_months:
        month_index = month_order.index(month)
        x_value = avg_monthly_rev.index[month_index]
        y_value = avg_monthly_rev.values[month_index]
        plt.vlines(x=x_value, ymin=17, ymax=y_value, linestyles="dotted", colors="red")

    # Shade the regions between vertical lines
    for i in range(1, len(highlight_months), 2):
        start_month = highlight_months[i - 1]
        end_month = highlight_months[i]
        start_index = month_order.index(start_month)
        end_index = month_order.index(end_month)
        plt.fill_between(
            avg_monthly_rev.index[start_index : end_index + 1],
            17,
            avg_monthly_rev.values[start_index : end_index + 1],
            alpha=0.2,
            color="gray",
        )

    plt.xlabel("Month")
    plt.ylabel("Average Revenue (£)")
    plt.title("Average Monthly Revenue")
    plt.xticks(rotation=45)  # Rotate x-axis tick labels for better visibility
    plt.legend()
    plt.show()


def train_test_split_time_series(
    df: pd.DataFrame,
    split_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time series DataFrame into training and testing sets.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with a date/time index.
    - split_ratio (float): The ratio for the training set. Default is 0.7.

    Returns:
    - train_set (pd.DataFrame): Training data with 'invoice_date' and 'revenue' columns.
    - test_set (pd.DataFrame): Test data with 'invoice_date' and 'revenue' columns.
    """
    # Calculate the total data size
    total_data_size = len(df)

    # Calculate the index for the split point
    split_index = int(split_ratio * total_data_size)

    # Split the DataFrame into training and testing sets
    train_set = df[['invoice_date', 'revenue']].iloc[:split_index]
    test_set = df[['invoice_date', 'revenue']].iloc[split_index:]

    return train_set, test_set


def plot_cluster_revenues(dataframes: List[pd.DataFrame], title: str):
    """
    Plot revenues for a list of DataFrames in a 2x2 grid.

    Parameters:
    - dataframes (List[pd.DataFrame]): List of ordered cluster DataFrames containing revenue data.
    - title (str): Super title for all subplots.

    Returns:
    None
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title.title(), fontsize=16)

    cluster_titles = [
        "Budget-Conscious Home Decor Shoppers",
        "Garden and Decor Enthusiasts",
        "Frequent Shopper/Top Customers",
        "High-End Shoppers with Moderate Value"
    ]

    for i, df in enumerate(dataframes):
        # Set df index to 'invoice_date' if it isn't already
        if df.index.name != 'invoice_date':
            df.set_index('invoice_date', inplace=True)
        row, col = i // 2, i % 2
        ax = axes[row, col]

        # Plot weekly revenue data
        ax.plot(df.index, df['revenue'], label='Weekly Revenue')
        ax.set_title(f'Revenue for {cluster_titles[i]}')  # Set the title based on the cluster_titles list
        ax.set_xlabel('Date')
        ax.set_ylabel('Revenue (£)')

        # Tilt the x-axis ticks for better readability
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()



def create_weekly_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create weekly training and test sets from DataFrames.

    Parameters:
    - train_df (pd.DataFrame): The training DataFrame containing time series data.
    - test_df (pd.DataFrame): The test DataFrame containing time series data.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Weekly training and test sets as DataFrames with 'invoice_date' index and 'revenue' column.
    """
    # Reset the index for training and test DataFrames to 'invoice_date' time series if not already
    if train_df.index.name != 'invoice_date':
        train_df.set_index('invoice_date', inplace=True)
    train_df_weekly = train_df[['revenue']].resample('W').sum().fillna(0)

    if test_df.index.name != 'invoice_date':
        test_df.set_index('invoice_date', inplace=True)
    test_df_weekly = test_df[['revenue']].resample('W').sum().fillna(0)

    return train_df_weekly, test_df_weekly


def adf_test_clusters(clusters: List[pd.Series]) -> pd.DataFrame:
    """
    Perform Augmented Dickey-Fuller test on time series data for multiple clusters.

    Parameters:
    - clusters (List[pd.Series]): A list of time series data for each cluster.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'Cluster', 'ADF Statistic', and 'p-value'.
    """
    results = []
    
    for i, cluster in enumerate(clusters):
        result = adfuller(cluster)
        adf_statistic = result[0]
        p_value = result[1]
        
        results.append({'Cluster': i, 'ADF Statistic': adf_statistic, 'p-value': p_value})
    
    return pd.DataFrame(results).set_index('Cluster')


def plot_acf_pacf(time_series, cluster=None, figsize=(12, 4)):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of a time series.

    Parameters:
    - time_series (pd.Series): The time series data to plot.
    - cluster (int): The cluster number. If provided, it's included in the plot titles.
    - figsize (tuple): The figure size in inches (width, height). Default is (12, 4).

    Returns:
    None
    """

    if cluster is not None:
        super_title = f"ACF and PACF for Cluster {cluster}"
    else:
        super_title = "ACF and PACF"

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(super_title, fontsize=16, y=1.05)  # Adjust the y-coordinate for more space

    # Plot ACF
    plot_acf(time_series, ax=axes[0])
    axes[0].set_title(f'Autocorrelation Function (ACF)', y=1.02)  # Adjust the y-coordinate

    # Plot PACF
    plot_pacf(time_series, ax=axes[1])
    axes[1].set_title(f'Partial Autocorrelation Function (PACF)', y=1.02)  # Adjust the y-coordinate

    plt.show()


def plot_residuals_arima(
    arima_model_fit,
    cluster: int = None,
):
    """
    Plot residuals and residual density along with ACF and PACF of residuals for an ARIMA model.

    Parameters:
    - arima_model_fit: Fitted ARIMA model.
    - cluster (int): Cluster number for the super title. Default is None.

    Returns:
    None
    """
    residuals = arima_model_fit.resid[1:]
    
    if cluster is not None:
        super_title = f"Residual Plots for Cluster {cluster}"
    else:
        super_title = "Residual Plots"
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(super_title, fontsize=16)  # Set the super title
    
    residuals.plot(title='Residuals', ax=ax[0, 0])
    ax[0, 0].set_xlabel('Invoice Date')
    residuals.plot(title='Density', kind='kde', ax=ax[0, 1])

    plot_acf(residuals, ax=ax[1, 0])
    ax[1, 0].set_title('Autocorrelation Function (ACF) - Residuals')

    plot_pacf(residuals, ax=ax[1, 1])
    ax[1, 1].set_title('Partial Autocorrelation Function (PACF) - Residuals')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def plot_manual_auto_forecasts(cluster2_weekly_test, cluster2_weekly_train, cluster2_weekly, arima_model_fit, auto_arima, steps_ahead=1, title="Cluster 2 Weekly Revenue Forecast"):
    """
    Plot manual and auto predictions together with the actual series, including the test set.

    Parameters:
    - cluster2_weekly_test (pd.Series): Test data.
    - cluster2_weekly_train (pd.Series): Training data.
    - cluster2_weekly (pd.DataFrame): Weekly data with forecasted columns.
    - arima_model_fit: Trained ARIMA model.
    - auto_arima: Trained auto-ARIMA model.
    - steps_ahead (int): Number of future steps to forecast. Default is 1.
    - title (str): Title for the plot. Default is "Cluster 2 Weekly Revenue Forecast".

    Returns:
    None
    """
    # Manual forecast
    forecast_test = arima_model_fit.forecast(len(cluster2_weekly_test))
    cluster2_weekly['forecast_manual'] = [None] * len(cluster2_weekly_train) + list(forecast_test)

    # Create a date range for future predictions
    last_date = cluster2_weekly_test.index[-1]
    future_dates = pd.date_range(start=last_date, periods=steps_ahead + 2)

    # Predict future data using ARIMA models
    future_manual_forecast = arima_model_fit.predict(start=len(cluster2_weekly_test), end=len(cluster2_weekly_test) + steps_ahead - 1)
    # Slice the forecast to exclude the last date
    future_manual_forecast = future_manual_forecast[:-1]

    # Plot manual and auto forecasts
    plt.figure(figsize=(10, 6))

    # Plot train and test data
    cluster2_weekly_train['revenue'].plot(label='Train Data')
    cluster2_weekly_test['revenue'].plot(label='Test Data')
    
    cluster2_weekly['forecast_manual'].plot(label='Manual Forecast', linestyle='--')
    
    # Auto forecast
    forecast_test_auto = auto_arima.predict(n_periods=len(cluster2_weekly_test))
    cluster2_weekly['forecast_auto'] = [None] * len(cluster2_weekly_train) + list(forecast_test_auto)
    cluster2_weekly['forecast_auto'].plot(label='Auto Forecast', linestyle='--')

    plt.title(title)
    plt.xlabel("Invoice Date")
    plt.ylabel('Revenue (£)')
    plt.legend()
    plt.show()
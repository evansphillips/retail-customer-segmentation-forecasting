import pandas as pd
import matplotlib.pyplot as plt
import re
import string
from typing import Optional
from transformers import BertModel, BertTokenizer
from typing import List, Tuple


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
        colors = [
        'gold',
        'skyblue',
        'lightcoral',
        'lightgreen',
        'lightpink',
        'lightblue',
        'lightyellow',
        'lightseagreen',
        'lightgray',
        'lightcyan',
        'lightsalmon',
        'lightcoral',
        ]
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


def preprocess_text(text: str) -> str:
    """
    Preprocess a text by removing punctuation, optionally removing stopwords, and lemmatizing.

    Parameters:
    - text (str): The input text to be preprocessed.

    Returns:
    str: The preprocessed text as a string.
    """
    # Convert to lowercase
    text = text.lower()

    # Define a list of metric system unit abbreviations
    metric_units = ['mm', 'cm', 'm', 'km', 'g', 'kg', 'ml', 'cl', 'dl', 'l']
    
    # Remove metric system units and their abbreviations
    text = re.sub(fr'(?<![a-zA-Z])({"|".join(metric_units)})(?![a-zA-Z])', '', text)

    # Remove extra whitespace and digits
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\d', '', text)  # Remove digits

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text




order_issue_words = [
    "cancellation",
    "missing",
    "error",
    "problem",
    "defect",
    "refund",
    "return",
    "discrepancy",
    "shortage",
    "delay",
    "damaged",
    "incorrect",
    "incomplete",
    "outofstock",
    "paymentissue",
    "billingerror",
    "wrongitem",
    "itemnotreceived",
    "deliveryissue",
    "qualityproblem",
    "pricingdiscrepancy",
    "shipmenterror",
    "lost",
    "doublecharge",
    "overcharge",
    "undercharge",
    "latedelivery",
    "cancelled",
    "rejected",
    "fraud",
    "wrongcolor",
    "exchange",
    "customization",
    "trackingproblem",
    "international",
    "addressissue",
    "trackingdelay",
    "customsissue",
    "importissue",
    "fulfillment",
    "creditcard",
    "recall",
    "scheduling",
    "productiondelay",
    "overstock",
    "supplierdelay",
    "inventorysurplus",
    "unprocessedpayment",
    "communication",
    "unresponsive",
    "verification",
    "discountcode",
    "outofprint",
    "assembly",
    "customsduty",
    "dutyfee",
    "verificationdelay",
    "addressvalidation",
    "notasdescribed",
    "trackingnumber",
    "recalled",
    "stolen",
    "returned",
    "customshold",
    "inspectiondelay",
    "importrestrictions",
    "holidaydelay",
    "damagedmerchandise",
    "factorydefect",
    "dataentry",
    "manualprocessing",
    "authentication",
    "shipmentrefusal",
    "inventorymislabeling",
    "labelingerror",
    "priceadjustment",
    "conversionissue",
    "datamigration",
    "lostpackage",
    "packagetheft",
    "mishandling",
    "missinglabel",
    "returnlabel",
    "routingerror",
    "packingerror",
    "chargeback",
    "rerouting",
    "cancellationerror",
    "renewalfailure",
    "refusal",
    "intercept",
    "reimbursement",
    "processingdelay",
    "resolution",
    "timeframeviolation",
    "multipleprocessing",
    "centererror",
    "reconciliation",
    "formatting",
    "paymenterror",
    "unfulfilled",
    "reversal",
    "expiration",
    "canceled",
    "missingcomponents",
    "substitution",
    "overcharge",
    "discounterror",
    "unauthorized",
    "failedtransaction",
    "chargeerror",
    "dataerror",
    "authorization",
    "unsuccessful",
    "rejectedpayment",
    "duplicateorder",
    "authorizationfailure",
    "stockout",
    "creditcarderror",
    "unavailable",
    "stockissue",
    "billedincorrectly",
    "unpaid",
    "paymentfailure",
    "duplicatecharge",
    "transactionerror",
    "unauthorizedcharge",
    "unverified",
    "unrecognizedtransaction",
    "duplicatebilling",
    "unreceived",
    "unconfirmed",
    "declined",
    "failedpayment",
    "creditcardissue",
    "unprocessedorder",
    "duplicatebilling",
    "shipmentproblem",
    "defective",
    "incorrectsize",
    "returnrequest",
    "shippingerror",
    "invoiceerror",
    "unshipped",
    "unfulfilledrequest",
    "canceledshipment",
    "miscommunication",
    "unsent",
    "unsatisfied",
    "stockmismatch",
    "bankerror",
    "undelivered",
    "unattended",
    "unsupportedpayment",
    "transactiondeclined",
    "nonreceipt",
    "undeliverable",
    "unavailableitem",
    "unfulfilledshipment",
    "invoicemistake",
    "processingerror",
    "unsatisfactory",
    "qualityissue",
    "qualitycontrol",
    "errornotification",
    "overbilling",
    "overdraft",
    "orderingerror",
    "fraudulentactivity",
    "stockreturn",
    "wronginvoice",
    "unverifiedorder",
    "shippingmistake",
    "customerror",
    "paymentdiscrepancy",
    "refundproblem",
    "overdelivery",
    "missingdelivery",
    "missingshipment",
    "missingitem",
    "missingproduct",
    "missingparcel",
    "missingpackage",
    "missinggoods",
    "missingstock",
    "missinginventory",
    "missingmerchandise",
    "missingcommodity",
    "missinggoods",
    "missingunit",
    "missingcarton",
    "missinglot",
    "missingset",
    "missingbatch",
    "missingshipment",
    "missingorder",
    "missingconsignment",
    "missingcargo",
    "missingfreight",
    "missingload",
    "missingcontainer",
    "missingshipment",
    "missingdelivery",
    "missingdispatch",
    "missingtransport",
    "missingcourier",
    "missingpost",
    "missingcarrier",
    "missingfreight",
    "missingdelivery",
    "missingshipment",
    "missingorder",
    "missingconsignment",
    "missingparcel",
    "missingpackage",
    "missinggoods",
    "missingstock",
    "missinginventory",
    "missingmerchandise",
    "missingcommodity",
    "missinggoods",
    "missingunit",
    "missingcarton",
    "missinglot",
    "missingset",
    "missingbatch",
    "missingshipment",
    "missingorder",
    "missingconsignment",
    "missingcargo",
    "missingfreight",
    "missingload",
    "missingcontainer",
    "missingshipment",
    "missingdelivery",
    "missingdispatch",
    "missingtransport",
    "missingcourier",
    "missingpost",
    "missingcarrier",
    "missingfreight",
    "missingdelivery",
    "missingshipment",
    "missingorder",
    "missingconsignment",
    "missingparcel",
    "missingpackage",
    "missinggoods",
    "missingstock",
    "missinginventory",
    "missingmerchandise",
    "missingcommodity",
    "missinggoods",
    "missingunit",
    "missingcarton",
    "missinglot",
    "missingset",
    "missingbatch",
    "missingshipment",
    "missingorder",
    "missingconsignment",
    "missingcargo",
    "missingfreight",
    "missingload",
    "missingcontainer",
    "missingshipment",
    "missingdelivery",
    "missingdispatch",
    "missingtransport",
    "missingcourier",
    "missingpost",
    "missingcarrier",
    "missingfreight",
    "missingdelivery",
    "missingshipment",
    "missingorder",
    "missingconsignment",
    "missingparcel",
    "missingpackage",
    "missinggoods",
    "missingstock",
    "missinginventory",
    "missingmerchandise",
    "missingcommodity",
    "missinggoods"]
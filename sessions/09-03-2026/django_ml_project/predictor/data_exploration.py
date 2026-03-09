import pandas as pd

# Data Exploration
def dataset_exploration(df):
    table_html = df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
        index=False,
    )
    return table_html

# Data description
def data_exploration(df):
    # Using describe() for statistical analysis as implied by the dashboard section
    table_html = df.describe().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
    )
    return table_html

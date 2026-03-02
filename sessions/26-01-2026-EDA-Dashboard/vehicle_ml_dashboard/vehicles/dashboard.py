import plotly.express as px
import plotly.offline as opy
import plotly.graph_objects as go

import pandas as pd


def frequency_table(df):
    """Generate a one-way frequency table for manufacturers."""
    # Simple counts
    manufacturer_counts = df['manufacturer'].value_counts().reset_index()
    manufacturer_counts.columns = ['Manufacturer', 'Count']

    # Convert to HTML using the correct method name: .to_html()
    table_html = manufacturer_counts.to_html(
        classes="table table-bordered table-striped table-sm",
        float_format='%.2f',
        justify='center'
    )
    return table_html
def sales_visualization_table(df):
    df['profit']=df['selling_price']-df['wholesale_price']
    table_html= df.groupby(['manufacturer','transmission','fuel_type']).agg({
        "profit":"sum",
        "selling_price":['sum','count'],
        "wholesale_price":'sum'
    }).to_html(
        classes="table table-bordered table-striped table-sm",
        float_format='%.2f',
        justify='center'
    )
    return table_html
def cross_tabulation_visualization_table(df):
    table_html=pd.crosstab(df['manufacturer'],df['transmission'],margins=True).to_html(
        classes="table table-bordered table-striped table-sm",
        float_format='%.2f',
        justify='center'
    )
    return table_html
def multi_dimensional_tabulation_visualization_table(df):
    table_html=pd.crosstab([df['manufacturer'],df['body_type']],[df['engine_type'],df['transmission']],margins=True).to_html(
    classes="table table-bordered table-striped table-sm",
    float_format='%.2f',
    justify='center'
    )
    return table_html
def cross_tabulation_with_more_details_visualization_table(df):
    table_html=pd.crosstab(df['manufacturer'],df['body_type'],values=df['selling_price'],aggfunc='sum',margins=True).to_html(
        classes="table table-bordered table-striped table-sm",
        float_format='%.2f',
        justify='center'
    )
    return table_html



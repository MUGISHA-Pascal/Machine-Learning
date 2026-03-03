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
def frequency_bar_chart(df):
    """Generate a bar chart for manufacturer frequencies."""
    manufacturer_counts = df['manufacturer'].value_counts().reset_index()
    manufacturer_counts.columns = ['Manufacturer', 'Count']
    fig = px.bar(manufacturer_counts, x='Manufacturer', y='Count', title='Manufacturer Frequency')
    return opy.plot(fig, output_type='div')

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
def cross_tabulation_with_lambda_visualization_table(df):
    def price_range(x):
        return x.max()-x.min()
    price_range.__name__='price range'
    table_html=pd.crosstab(df['manufacturer'],df['body_type'],values=df['selling_price'],aggfunc=[price_range,'sum'],margins=True).to_html(
        classes="table table-bordered table-striped table-sm",
        float_format='%.2f',
        justify='center'
    )
    return table_html
def pivot_table(df):
    table_html=pd.pivot_table(df,index=['manufacturer','body_type'],values=['selling_price'],aggfunc=['sum','count']).to_html(
        classes="table table-bordered table-striped table-sm",
        float_format='%.2f',
        justify='center'
    )
    return table_html
def sales_visualization_with_sunburst_chart(df,height=800):
    fig=px.sunburst(df,path=['manufacturer','fuel_type','body_type'],values='selling_price')
    fig.update_traces(textinfo='label+value')
    fig.update_layout(height=height)
    return opy.plot(fig,output_type='div')
def sales_visualization_with_tree_map_chart(df,height=800):
    fig=px.treemap(df,path=['manufacturer','fuel_type','body_type'],values='selling_price')
    fig.update_traces(textinfo='label+value')
    fig.update_layout(height=height)
    return opy.plot(fig,output_type='div')
def sales_visualization_with_icicle_chart(df,height=800):
    fig=px.icicle(df,path=['manufacturer','fuel_type','body_type'],values='selling_price')
    fig.update_traces(textinfo='label+value')
    fig.update_layout(height=height)
    return opy.plot(fig,output_type='div')
def client_distribution_by_country_map(df):
    country_counts = df['client_country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    fig = px.choropleth(country_counts, locations='Country', locationmode='country names', color='Count', title='Client Distribution by Country')
    fig.update_layout(height=600)
    return opy.plot(fig, output_type='div')
    
def sales_distribution_by_country_map(df):
    sales_by_country = df.groupby('client_country')['selling_price'].sum().reset_index()
    sales_by_country.columns = ['Country', 'Sales']
    fig = go.Figure(data=go.Choropleth(
        locations=sales_by_country['Country'],
        locationmode='country names',
        z=sales_by_country['Sales'],
        text=sales_by_country['Country'],
        colorscale='Blues',
        colorbar_title='Sales',
    ))
    fig.add_trace(go.Scattergeo(
        locations=sales_by_country['Country'],
        locationmode='country names',
        mode='text',
        text=sales_by_country.apply(lambda row: f"{row['Country']}: {row['Sales']}", axis=1),
        textposition='top center',
        showlegend=False
    ))
    fig.update_layout(
        title_text='Sales Distribution by Country',
        geo=dict(showframe=False, showcoastlines=False),
        height=600
    )
    return opy.plot(fig, output_type='div')
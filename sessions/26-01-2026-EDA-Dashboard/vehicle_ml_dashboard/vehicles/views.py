import pandas as pd
from django.shortcuts import render
from .dashboard import (
    frequency_table,
    frequency_bar_chart,
    sales_visualization_table,
    cross_tabulation_visualization_table,
    multi_dimensional_tabulation_visualization_table,
    cross_tabulation_with_more_details_visualization_table,
    cross_tabulation_with_lambda_visualization_table,
    pivot_table,
    sales_visualization_with_sunburst_chart,
    sales_visualization_with_tree_map_chart,
    sales_visualization_with_icicle_chart
)

def dashboard_view(request):
    """Main dashboard view that loads vehicle data and renders charts."""
    queryset = pd.read_csv("dummy_data/vehicles_data_1000.csv")
    df = pd.DataFrame(queryset)

    return render(request, "vehicles/index.html", {
        "frequency_table": frequency_table(df),
        "frequency_bar_chart": frequency_bar_chart(df),
        "sales_table": sales_visualization_table(df),
        "sales_sunburst_chart": sales_visualization_with_sunburst_chart(df),
        "sales_tree_map_chart": sales_visualization_with_tree_map_chart(df),
        "sales_icicle_chart": sales_visualization_with_icicle_chart(df),
        "cross_tabulation_table": cross_tabulation_visualization_table(df),
        "multi_dimensional_table": multi_dimensional_tabulation_visualization_table(df),
        "cross_tabulation_details_table": cross_tabulation_with_more_details_visualization_table(df),
        "cross_tabulation_lambda_table": cross_tabulation_with_lambda_visualization_table(df),
        "pivot_table": pivot_table(df)
    })

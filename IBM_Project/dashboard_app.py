from pyspark.sql import SparkSession
import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.express as px
import pandas as pd
import us  # Library to map state names to abbreviations

# Create a dictionary to map full state names to abbreviations
state_abbr_map = {state.name: state.abbr for state in us.states.STATES}

# -----------------------------
# Step 1: Initialize Spark & Load Processed Data
# -----------------------------
spark = SparkSession.builder.getOrCreate()
processed_data_path = "processed_data_bucket/processed_customer_purchase_behavior.csv"
processed_df = spark.read.option("header", "true").option(
    "inferSchema", "true").csv(processed_data_path)
data_pd = processed_df.toPandas()
# Load Predictions from Random Forest Model
predictions_path = "content\\predictions\\Random_Forest_predictions.parquet"
predictions_df = spark.read.parquet(predictions_path)

# Convert to Pandas for Visualization
predictions_pd = predictions_df.toPandas()

# Merge Predictions with Processed Data (Assuming 'Customer ID' exists in both)
if "Customer ID" in data_pd.columns and "Customer ID" in predictions_pd.columns:
    data_pd = data_pd.merge(predictions_pd, on="Customer ID", how="left")

# Verify Available Columns
print("Columns in Predictions Data:", predictions_pd.columns)

# -----------------------------
# Step 2: Utility Functions
# -----------------------------


def check_column(df, column):
    return column in df.columns


# Extract Unique Values for Filters
unique_categories = data_pd["Category"].unique(
) if check_column(data_pd, "Category") else []
unique_locations = data_pd["Location"].unique(
) if check_column(data_pd, "Location") else []
unique_payment_methods = data_pd["Payment Method"].unique(
) if check_column(data_pd, "Payment Method") else []
unique_seasons = data_pd["Season"].unique(
) if check_column(data_pd, "Season") else []

# Compute Summary Metrics
total_orders = len(data_pd)
total_customers = len(data_pd['Customer ID'].unique()) if check_column(
    data_pd, "Customer ID") else 0
total_sales = data_pd['Purchase Amount (USD)'].sum() if check_column(
    data_pd, "Purchase Amount (USD)") else 0
average_order_value = total_sales / total_orders if total_orders > 0 else 0

# -----------------------------
# Step 3: Dashboard Theme & Layout
# -----------------------------
theme_colors = {
    "background": "#121212",
    "text": "#EAEAEA",
    "primary": "#007BFF",
    "secondary": "#0056b3"
}

dropdown_style = {
    'width': '220px',
    'borderRadius': '5px',
    'color': 'black'
}


app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Needed for deployment

app.layout = html.Div(style={
    'backgroundColor': theme_colors['background'],
    'color': theme_colors['text'],
    'padding': '20px'
}, children=[
    html.H1("📊 Customer Purchase Analytics Dashboard", style={
        'textAlign': 'center', 'color': theme_colors['primary']}),

    # Summary Statistics
    html.Div(className="summary-container", children=[
        html.Div(className="metric-box",
                 children=[html.H4("Total Customers"), html.P(f"{total_customers:,}")]),
        html.Div(className="metric-box",
                 children=[html.H4("Total Orders"), html.P(f"{total_orders:,}")]),
        html.Div(className="metric-box",
                 children=[html.H4("Total Sales (USD)"), html.P(f"${total_sales:,.2f}")]),
        html.Div(className="metric-box", children=[html.H4(
            "Average Order Value (AOV)"), html.P(f"${average_order_value:,.2f}")])
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}),

    # Filters with Improved Contrast
    html.Div(className="filter-container", style={
        'display': 'flex',
        'gap': '10px',
        'flexWrap': 'wrap',
        'justifyContent': 'center',
        'padding': '10px',
        'backgroundColor': '#f0f0f0',  # Light grey for contrast
        'borderRadius': '8px'  # Smooth edges
    }, children=[
        dcc.Dropdown(
            id="category-filter",
            options=[{'label': cat, 'value': cat}
                     for cat in unique_categories],
            multi=True,
            placeholder="Category",
            style=dropdown_style
        ),
        dcc.Dropdown(
            id="location-filter",
            options=[{'label': loc, 'value': loc}
                     for loc in unique_locations],
            multi=True,
            placeholder="Location",
            style=dropdown_style
        ),
        dcc.Dropdown(
            id="payment-filter",
            options=[{'label': pay, 'value': pay}
                     for pay in unique_payment_methods],
            multi=True,
            placeholder="Payment Method",
            style=dropdown_style
        ),
        dcc.Dropdown(
            id="season-filter",
            options=[{'label': season, 'value': season}
                     for season in unique_seasons],
            multi=True,
            placeholder="Season",
            style=dropdown_style
        ),

        html.Button("🔄 Clear Filters", id="clear-filters-btn", style={
            'backgroundColor': theme_colors['secondary'],
            'color': 'white',
            'padding': '8px 15px',
            'borderRadius': '5px',
            'cursor': 'pointer'
        }),
        html.Button("📥 Download CSV Report", id="download-btn-csv", style={
            'backgroundColor': theme_colors['primary'],
            'color': 'white',
            'padding': '8px 15px',
            'borderRadius': '5px',
            'cursor': 'pointer'
        }),
        dcc.Download(id="download")
    ]),

    # Tabs
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label="📊 Sales Overview", value="tab-1"),
        dcc.Tab(label="💳 Payment & Shipping Insights", value="tab-2"),
        dcc.Tab(label="📍 Geographic Trends", value="tab-3"),
        dcc.Tab(label="👥 Customer Segmentation", value="tab-4"),
    ]),
    html.Div(id="tabs-content")
])


# -----------------------------
# Step 4: Callbacks
# -----------------------------

# Callback to Clear Filters


@app.callback(
    [Output("category-filter", "value"),
     Output("location-filter", "value"),
     Output("payment-filter", "value"),
     Output("season-filter", "value")],
    Input("clear-filters-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_filters(n_clicks):
    return [], [], [], []

# Callback to Update Charts


@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs", "value"),
     Input("category-filter", "value"),
     Input("location-filter", "value"),
     Input("payment-filter", "value"),
     Input("season-filter", "value")]
)
def update_content(tab, selected_categories, selected_locations, selected_payments, selected_seasons):
    filtered_data = data_pd.copy()

    if selected_categories:
        filtered_data = filtered_data[filtered_data["Category"].isin(
            selected_categories)]
    if selected_locations:
        filtered_data = filtered_data[filtered_data["Location"].isin(
            selected_locations)]
    if selected_payments:
        filtered_data = filtered_data[filtered_data["Payment Method"].isin(
            selected_payments)]
    if selected_seasons:
        filtered_data = filtered_data[filtered_data["Season"].isin(
            selected_seasons)]

    if tab == "tab-1":  # 📊 Sales Overview
        fig_sales_sunburst = px.sunburst(
            filtered_data,
            path=["Category"],
            values="Purchase Amount (USD)",
            title="💰 Sales by Category",
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Prism,  # Vibrant colors
            template="plotly_white"
        )

        fig_sales_bar = px.bar(
            filtered_data.groupby("Category")[
                "Purchase Amount (USD)"].sum().reset_index(),
            x="Category",
            y="Purchase Amount (USD)",
            title="📊 Total Sales per Category",
            text_auto=True,
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        fig_sales_sunburst.update_layout(
            title_font_size=20,
            title_x=0.5,
            xaxis_title="Category",
            yaxis_title="Total Sales (USD)",
            plot_bgcolor="white",
            paper_bgcolor=theme_colors['background'],
            font=dict(family="Arial, sans-serif", size=12,
                      color=theme_colors['text']),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#ddd"),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return html.Div([
            html.Div([dcc.Graph(figure=fig_sales_sunburst)], style={
                     'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_sales_bar)], style={
                     'width': '50%', 'display': 'inline-block'})
        ])

    elif tab == "tab-2":  # 💳 Payment & Shipping Insights
        fig_payment_pie = px.pie(
            filtered_data,
            names="Payment Method",
            title="💳 Payment Methods",
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig_payment_bar = px.bar(
            filtered_data.groupby("Payment Method")[
                "Purchase Amount (USD)"].sum().reset_index(),
            x="Payment Method",
            y="Purchase Amount (USD)",
            title="💰 Total Sales by Payment Method",
            text_auto=True,
            color="Payment Method",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        return html.Div([
            html.Div([dcc.Graph(figure=fig_payment_pie)], style={
                     'width': '50%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig_payment_bar)], style={
                     'width': '50%', 'display': 'inline-block'})
        ])

    elif tab == "tab-3":  # 📍 Geographic Trends
        import us  # Library to map state names to abbreviations

        # Create a dictionary to map full state names to abbreviations
        state_abbr_map = {state.name: state.abbr for state in us.states.STATES}

        # Apply mapping
        filtered_data["State Abbr"] = filtered_data["Location"].map(
            state_abbr_map)

        # Plot the updated map
        fig_location = px.choropleth(
            filtered_data,
            locations="State Abbr",  # Use state abbreviations now
            locationmode="USA-states",
            color="Purchase Amount (USD)",
            title="🌎 Sales by Location",
            color_continuous_scale=px.colors.sequential.Plasma
        )

        fig_location.update_layout(
            geo=dict(bgcolor=theme_colors['background']),
            paper_bgcolor=theme_colors['background'],
            font=dict(color=theme_colors['text'])
        )

        return html.Div([dcc.Graph(figure=fig_location)])
    elif tab == "tab-4":  # 👥 Customer Segmentation

        # 🔵 1. Pie Chart: High-Value Customer Predictions
        if "prediction" in predictions_pd.columns:
            fig_predicted_customers = px.pie(
                predictions_pd.fillna({"prediction": "Unknown"}),
                names="prediction",
                title="🔮 Predicted High-Value Customers",
                labels={"prediction": "Customer Type"},
                color_discrete_sequence=px.colors.qualitative.Safe
            )
        else:
            fig_predicted_customers = px.pie(
                title="Predictions Data Not Available")

        # 🟠 2. Histogram: Actual vs Predicted Distribution
        if "label" in predictions_pd.columns and "prediction" in predictions_pd.columns:
            filtered_df = predictions_pd[[
                "label", "prediction"]].fillna("Unknown")

            fig_actual_vs_predicted = px.histogram(
                filtered_df.melt(value_vars=["label", "prediction"]),
                x="value",
                title="✅ Actual vs Predicted Customer Segments",
                labels={"value": "Customer Segment"},
                barmode="overlay",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
        else:
            fig_actual_vs_predicted = px.histogram(
                title="Actual vs Predicted Data Not Available")

        # 🟢 3. Probability Distribution (Confidence of Predictions)
        if "probability" in predictions_pd.columns:
            fig_probabilities = px.histogram(
                predictions_pd,
                x=predictions_pd["probability"].apply(
                    lambda x: x[1]),  # Extract probability of class 1
                title="📊 Model Confidence Distribution",
                labels={"x": "Prediction Confidence"},
                nbins=20,
                color_discrete_sequence=["#FFA07A"]
            )
        else:
            fig_probabilities = px.histogram(
                title="Model Confidence Data Not Available")

        # 🔳 Layout: Display prediction-based insights
        return html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'justifyContent': 'space-around'}, children=[
            html.Div([dcc.Graph(figure=fig_predicted_customers)],
                     style={'width': '32%'}),
            html.Div([dcc.Graph(figure=fig_actual_vs_predicted)],
                     style={'width': '32%'}),
            html.Div([dcc.Graph(figure=fig_probabilities)],
                     style={'width': '32%'})
        ])

    # elif tab == "tab-4":  # 👥 Customer Segmentation

    #     # 🔵 1. Scatter Plot: Frequency vs. Spending
    #     fig_customer_scatter = px.scatter(
    #         filtered_data,
    #         x="Frequency of Purchases",
    #         y="Purchase Amount (USD)",
    #         color="Subscription Status",
    #         size="Purchase Amount (USD)",
    #         title="🛍️ Spending vs. Purchase Frequency",
    #         color_discrete_sequence=px.colors.qualitative.Bold
    #     )

    #     # 🟠 2. Pie Chart: Subscription Status
    #     fig_customer_pie = px.pie(
    #         filtered_data,
    #         names="Subscription Status",
    #         title="🔄 Subscription Distribution",
    #         color_discrete_sequence=px.colors.qualitative.Safe
    #     )

    #     # 🟢 3. Bar Chart: Age Group Distribution
    #     filtered_data["Age Group"] = pd.cut(
    #         filtered_data["Age"],
    #         bins=[18, 25, 35, 50, 65, 80],
    #         labels=["18-25", "26-35", "36-50", "51-65", "65+"]
    #     )

    #     fig_customer_age = px.bar(
    #         filtered_data["Age Group"].value_counts().reset_index(),
    #         x="Age Group",
    #         y="count",
    #         title="👥 Age Group Distribution",
    # #         text_auto=True,
    # #         color_discrete_sequence=px.colors.qualitative.Pastel
    # #     )

    #     # 🔳 Layout: Display all 3 charts side by side
    #     return html.Div(style={'display': 'flex', 'flex-wrap': 'wrap', 'justifyContent': 'space-around'}, children=[
    #         html.Div([dcc.Graph(figure=fig_customer_scatter)],
    #                  style={'width': '48%'}),
    #         html.Div([dcc.Graph(figure=fig_customer_pie)],
    #                  style={'width': '48%'}),
    #         html.Div([dcc.Graph(figure=fig_customer_age)],
    #                  style={'width': '48%'})
    #     ])


# Callback to Handle CSV Download


@app.callback(
    Output("download", "data"),
    Input("download-btn-csv", "n_clicks"),
    prevent_initial_call=True
)
def download_csv_report(n_clicks):
    return dcc.send_data_frame(data_pd.to_csv, filename="customer_purchase_report.csv")


# -----------------------------
# Step 5: Run the App
# -----------------------------
if __name__ == '__main__':
    app.run_server(debug=True)

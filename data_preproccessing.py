# data_preprocessing.py

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For machine learning
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# For interactive dashboard
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

# For trendline
import statsmodels.api as sm

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings('ignore')


# Function to convert percentage strings to floats
def convert_percentage(value):
    try:
        return float(value.strip('%'))
    except:
        return np.nan


def preprocess_data():
    # Load the dataset
    try:
        data = pd.read_csv('champions.csv')
        print("Initial Data:")
        print(data.head())
    except FileNotFoundError:
        print("Error: The file 'champions.csv' was not found in the current directory.")
        exit()

    # Drop rows where key statistics are missing
    key_columns = ['Wins', 'Losses', 'Winrate', 'KDA', 'CSM', 'DPM', 'GPM', 'CSD@15', 'GD@15', 'XPD@15']
    initial_shape = data.shape
    data.dropna(subset=key_columns, inplace=True)
    print(
        f"\nData after dropping rows with missing values: {data.shape[0]} rows remaining (Dropped {initial_shape[0] - data.shape[0]} rows).")

    # Convert 'Presence' from string percentage to float
    data['Presence'] = data['Presence'].apply(convert_percentage)

    # Convert 'Winrate' from string percentage to float
    data['Winrate'] = data['Winrate'].apply(convert_percentage)

    # Handle 'KDA' which is now a single numerical value
    data['KDA_numeric'] = pd.to_numeric(data['KDA'], errors='coerce')

    # Inspect the 'KDA' column and 'KDA_numeric' after conversion
    print("\nUnique values in 'KDA' after dropping rows:")
    print(data['KDA'].unique())

    print("\nKDA_numeric after conversion:")
    print(data['KDA_numeric'].head())

    # Drop the original 'KDA' column as we've created a numeric version
    data.drop(['KDA'], axis=1, inplace=True)

    # Update the list of numerical columns after conversion
    numerical_columns = ['Picks', 'Bans', 'Presence', 'Wins', 'Losses', 'Winrate',
                         'CSM', 'DPM', 'GPM', 'CSD@15', 'GD@15', 'XPD@15', 'KDA_numeric']

    # Convert all numerical columns to numeric types and handle any residual NaNs
    for col in numerical_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Check for any remaining NaNs in numerical columns
    print("\nChecking for NaNs in numerical columns after type conversion:")
    print(data[numerical_columns].isnull().sum())

    # Initialize an imputer to fill any remaining NaNs with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

    # Verify that there are no more NaNs in numerical columns
    print("\nMissing values after imputation:")
    print(pd.DataFrame(data[numerical_columns].isnull().sum()).T)

    # Feature Engineering

    # Calculate Total Games Played
    data['Total_Games'] = data['Wins'] + data['Losses']

    # Calculate Average Gold Difference per Minute
    data['Avg_GD_per_Min'] = data['GD@15'] / 15

    # Encode 'Champion' using Label Encoding
    label_encoder = LabelEncoder()
    data['Champion_encoded'] = label_encoder.fit_transform(data['Champion'])

    # Display the preprocessed data
    print("\nPreprocessed Data:")
    print(data.head())

    # Save the preprocessed data for future use
    data.to_csv('champions_preprocessed.csv', index=False)
    print("\nPreprocessed data saved to 'champions_preprocessed.csv'.")

    # Return the preprocessed data for further use if needed
    return data


# Define the Dash app outside the main execution block
app = Dash(__name__)
app.title = "League of Legends Champions Dashboard"
# app._favicon = ("favicon.ico")  # Not using to prevent FileNotFoundError

# Suppress callback exceptions
app.config.suppress_callback_exceptions = True

# Load the preprocessed data
try:
    preprocessed_data = pd.read_csv('champions_preprocessed.csv')
except FileNotFoundError:
    print("Error: The file 'champions_preprocessed.csv' was not found. Please run the script to preprocess data first.")
    exit()

# List of numerical columns for visualization (excluding 'Total_Games' and 'Avg_GD_per_Min')
visualization_columns = ['Picks', 'Bans', 'Presence', 'Wins', 'Losses', 'Winrate',
                         'CSM', 'DPM', 'GPM', 'CSD@15',
                         'GD@15', 'XPD@15']

# Create options for the dropdown in the dashboard
key_features = ['DPM', 'GPM', 'CSM']  # Key features for scatter plots

if 'KDA_numeric' in preprocessed_data.columns:
    key_features.append('KDA_numeric')

dashboard_features = key_features  # Features used in scatter plots

# Create options for the dropdown
options = [{'label': col.replace('_', ' '), 'value': col} for col in dashboard_features]

# Define the layout of the dashboard, maintaining dark mode but leaving dropdowns in default styling
app.layout = html.Div([
    html.H1("League of Legends Champions Dashboard",
            style={'textAlign': 'center', 'font-family': 'Arial', 'color': '#ffffff'}),
    html.Div([
        html.Div([
            html.Label("Select Feature for Winrate Scatter Plot:", style={'font-weight': 'bold', 'color': '#ffffff'}),
            dcc.Dropdown(
                id='feature-dropdown',
                options=options,
                value=dashboard_features[0],
                style={'width': '100%'},
                # Optional: To keep default styling, do not set 'backgroundColor' and 'color'
                # If you prefer light background with dark text, uncomment the line below
                # style={'width': '100%', 'backgroundColor': '#ffffff', 'color': '#000000'},
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-right': '2%'}),
        html.Div([
            html.Label("Select Champion:", style={'font-weight': 'bold', 'color': '#ffffff'}),
            dcc.Dropdown(
                id='champion-dropdown',
                options=[{'label': champ, 'value': champ} for champ in sorted(preprocessed_data['Champion'].unique())],
                value=preprocessed_data['Champion'].unique()[0],
                style={'width': '100%'},
                # Optional: To keep default styling, do not set 'backgroundColor' and 'color'
                # If you prefer light background with dark text, uncomment the line below
                # style={'width': '100%', 'backgroundColor': '#ffffff', 'color': '#000000'},
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'float': 'right', 'padding-left': '2%'}),
    ], style={'padding': '10px 0'}),
    # Tabs for organizing plots
    dcc.Tabs([
        dcc.Tab(label='Individual Champion Stats', children=[
            html.Div([
                dcc.Graph(id='champion-bar-plot'),
            ], style={'padding': '20px'}),
            # Separator
            html.Hr(style={'borderColor': '#ffffff', 'margin': '40px 0'}),
            html.Div([
                dcc.Graph(id='champion-radar-plot'),
            ], style={'padding': '20px'}),
            # Separator
            html.Hr(style={'borderColor': '#ffffff', 'margin': '40px 0'}),
            html.Div([
                dcc.Graph(id='champion-scatter-plot'),
            ], style={'padding': '20px'}),
            # Separator
            html.Hr(style={'borderColor': '#ffffff', 'margin': '40px 0'}),
            html.Div([
                html.H3(id='champion-stats-text', style={'color': '#ffffff', 'whiteSpace': 'pre-line'}),
                dcc.Graph(id='champion-box-plot'),
            ], style={'padding': '20px'}),
        ], style={'backgroundColor': '#303030', 'color': '#ffffff'}),
        dcc.Tab(label='Generalized Dataset Insights', children=[
            html.Div([
                dcc.Graph(id='scatter-plot'),  # Winrate vs Selected Feature
            ], style={'padding': '20px'}),
            # Separator
            html.Hr(style={'borderColor': '#ffffff', 'margin': '40px 0'}),
            html.Div([
                dcc.Graph(id='feature-importance-bar'),
            ], style={'padding': '20px'}),
            # Separator
            html.Hr(style={'borderColor': '#ffffff', 'margin': '40px 0'}),
            html.Div([
                dcc.Graph(id='correlation-heatmap'),
            ], style={'padding': '20px'}),
            # Separator
            html.Hr(style={'borderColor': '#ffffff', 'margin': '40px 0'}),
            html.Div([
                dcc.Graph(id='winrate-histogram'),
            ], style={'padding': '20px'}),
            # Separator
            html.Hr(style={'borderColor': '#ffffff', 'margin': '40px 0'}),
            html.Div([
                dcc.Graph(id='metrics-box-plot'),
            ], style={'padding': '20px'}),
            # Separator
            html.Hr(style={'borderColor': '#ffffff', 'margin': '40px 0'}),
            html.Div([
                dcc.Graph(id='pair-plot'),
            ], style={'padding': '20px'}),
        ], style={'backgroundColor': '#303030', 'color': '#ffffff'}),
    ], style={'backgroundColor': '#4a4a4a', 'color': '#ffffff'}),
], style={'font-family': 'Arial', 'backgroundColor': '#303030'})

# Apply dark theme to graphs
graph_template = 'plotly_dark'


# Define callback to update scatter plot based on selected feature
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_scatter_plot(selected_feature):
    if selected_feature and selected_feature in preprocessed_data.columns:
        # Create scatter plot using Plotly Express
        fig = px.scatter(preprocessed_data, x=selected_feature, y='Winrate',
                         hover_data=['Champion'],
                         title=f'Winrate vs {selected_feature.replace("_", " ")}',
                         labels={selected_feature: selected_feature.replace('_', ' '), 'Winrate': 'Winrate (%)'},
                         trendline='ols',
                         template=graph_template)
        fig.update_traces(marker=dict(size=10,
                                      line=dict(width=1,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(paper_bgcolor='#303030', plot_bgcolor='#303030',
                          title_font=dict(color='#ffffff'),
                          font=dict(color='#ffffff'))
        return fig
    else:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No Feature Selected",
            paper_bgcolor='#303030',
            plot_bgcolor='#303030',
            title_font=dict(color='#ffffff'),
            font=dict(color='#ffffff')
        )
        return fig


# Define callback to update bar plot based on selected champion
@app.callback(
    Output('champion-bar-plot', 'figure'),
    Input('champion-dropdown', 'value')
)
def update_champion_bar_plot(selected_champion):
    if selected_champion and selected_champion in preprocessed_data['Champion'].values:
        # Filter data for the selected champion
        champ_data = preprocessed_data[preprocessed_data['Champion'] == selected_champion]
        if champ_data.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Champion Data Not Found",
                paper_bgcolor='#303030',
                plot_bgcolor='#303030',
                title_font=dict(color='#ffffff'),
                font=dict(color='#ffffff')
            )
            return fig

        # Select statistics to display
        stats = ['Winrate', 'DPM', 'GPM', 'CSM']
        if 'KDA_numeric' in champ_data.columns:
            stats.append('KDA_numeric')
        stats = [stat for stat in stats if stat in champ_data.columns]

        values = champ_data[stats].values[0]

        # Create bar plot using Plotly Graph Objects
        fig = go.Figure([go.Bar(x=stats, y=values, marker_color='indianred')])
        fig.update_layout(title=f'Statistics for {selected_champion}',
                          xaxis_title='Statistics',
                          yaxis_title='Values',
                          template=graph_template,
                          paper_bgcolor='#303030',
                          plot_bgcolor='#303030',
                          title_font=dict(color='#ffffff'),
                          font=dict(color='#ffffff'))
        return fig
    else:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No Champion Selected",
            paper_bgcolor='#303030',
            plot_bgcolor='#303030',
            title_font=dict(color='#ffffff'),
            font=dict(color='#ffffff')
        )
        return fig


# Define callback to update radar plot based on selected champion
@app.callback(
    Output('champion-radar-plot', 'figure'),
    Input('champion-dropdown', 'value')
)
def update_champion_radar_plot(selected_champion):
    if selected_champion and selected_champion in preprocessed_data['Champion'].values:
        # Filter data for the selected champion
        champ_data = preprocessed_data[preprocessed_data['Champion'] == selected_champion]
        if champ_data.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Champion Data Not Found",
                paper_bgcolor='#303030',
                plot_bgcolor='#303030',
                title_font=dict(color='#ffffff'),
                font=dict(color='#ffffff')
            )
            return fig

        # Select statistics to display
        stats = ['Winrate', 'DPM', 'GPM', 'CSM', 'KDA_numeric']
        stats = [stat for stat in stats if stat in champ_data.columns]

        # Normalize the values to be between 0 and 10
        scaler = MinMaxScaler(feature_range=(0, 10))
        stats_data = preprocessed_data[stats]
        scaler.fit(stats_data)
        normalized_values = scaler.transform(champ_data[stats])
        values = normalized_values[0].tolist()
        values += values[:1]  # Repeat the first value to close the radar chart

        theta = stats + [stats[0]]

        # Create radar chart, will be normalized now
        fig = go.Figure(
            data=go.Scatterpolar(r=values, theta=theta,
                                 fill='toself',
                                 name=selected_champion,
                                 marker_color='indianred')
        )
        fig.update_layout(
            polar=dict(
                bgcolor='#303030',
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    gridcolor='gray',
                    linecolor='gray'
                ),
                angularaxis=dict(gridcolor='gray', linecolor='gray')
            ),
            title=f'Normalized Radar Chart for {selected_champion}',
            showlegend=False,
            template=graph_template,
            paper_bgcolor='#303030',
            plot_bgcolor='#303030',
            title_font=dict(color='#ffffff'),
            font=dict(color='#ffffff')
        )
        return fig
    else:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No Champion Selected",
            paper_bgcolor='#303030',
            plot_bgcolor='#303030',
            title_font=dict(color='#ffffff'),
            font=dict(color='#ffffff')
        )
        return fig


# Define callback to update scatter plot for champion stats
@app.callback(
    Output('champion-scatter-plot', 'figure'),
    Input('champion-dropdown', 'value')
)
def update_champion_scatter_plot(selected_champion):
    if selected_champion and selected_champion in preprocessed_data['Champion'].values:
        # Example: DPM vs GPM for the selected champion
        champ_data = preprocessed_data[preprocessed_data['Champion'] == selected_champion]
        if champ_data.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Champion Data Not Found",
                paper_bgcolor='#303030',
                plot_bgcolor='#303030',
                title_font=dict(color='#ffffff'),
                font=dict(color='#ffffff')
            )
            return fig

        fig = px.scatter(champ_data, x='DPM', y='GPM',
                         title=f'DPM vs GPM for {selected_champion}',
                         labels={'DPM': 'Damage Per Minute', 'GPM': 'Gold Per Minute'},
                         hover_data=['Champion'],
                         template=graph_template)
        fig.update_traces(marker=dict(size=12,
                                      color='indianred',
                                      line=dict(width=1,
                                                color='DarkSlateGrey')))
        fig.update_layout(paper_bgcolor='#303030', plot_bgcolor='#303030',
                          title_font=dict(color='#ffffff'),
                          font=dict(color='#ffffff'))
        return fig
    else:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No Champion Selected",
            paper_bgcolor='#303030',
            plot_bgcolor='#303030',
            title_font=dict(color='#ffffff'),
            font=dict(color='#ffffff')
        )
        return fig


# Define callback to update box plot for champion stats
@app.callback(
    Output('champion-box-plot', 'figure'),
    Input('champion-dropdown', 'value')
)
def update_champion_box_plot(selected_champion):
    if selected_champion and selected_champion in preprocessed_data['Champion'].values:
        # Compare selected champion's stats against the entire dataset
        champ_data = preprocessed_data[preprocessed_data['Champion'] == selected_champion]
        if champ_data.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Champion Data Not Found",
                paper_bgcolor='#303030',
                plot_bgcolor='#303030',
                title_font=dict(color='#ffffff'),
                font=dict(color='#ffffff')
            )
            return fig

        stats = ['Winrate', 'DPM', 'GPM', 'CSM', 'KDA_numeric']
        stats = [stat for stat in stats if stat in champ_data.columns]

        # Initialize subplot with one row and len(stats) columns
        fig = make_subplots(rows=1, cols=len(stats),
                            subplot_titles=[stat.replace('_', ' ') for stat in stats])

        for i, stat in enumerate(stats):
            # Overall distribution
            fig.add_trace(go.Box(y=preprocessed_data[stat], name='All Champions',
                                 boxpoints='outliers', marker_color='lightblue', showlegend=False), row=1, col=i + 1)
            # Selected champion
            fig.add_trace(go.Box(y=champ_data[stat], name=selected_champion,
                                 boxpoints='outliers', marker_color='indianred', showlegend=False), row=1, col=i + 1)

        fig.update_layout(title=f'Comparison of {selected_champion} Stats with All Champions',
                          showlegend=False,
                          template=graph_template,
                          paper_bgcolor='#303030',
                          plot_bgcolor='#303030',
                          title_font=dict(color='#ffffff'),
                          font=dict(color='#ffffff'))

        return fig
    else:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="No Champion Selected",
            paper_bgcolor='#303030',
            plot_bgcolor='#303030',
            title_font=dict(color='#ffffff'),
            font=dict(color='#ffffff')
        )
        return fig


# Define callback to update champion stats text
@app.callback(
    Output('champion-stats-text', 'children'),
    Input('champion-dropdown', 'value')
)
def update_champion_stats_text(selected_champion):
    if selected_champion and selected_champion in preprocessed_data['Champion'].values:
        champ_data = preprocessed_data[preprocessed_data['Champion'] == selected_champion]
        stats = ['Winrate', 'DPM', 'GPM', 'CSM', 'KDA_numeric']
        stats = [stat for stat in stats if stat in champ_data.columns]
        values = champ_data[stats].values[0]

        # Create a list of stat strings
        stat_strings = [f"{stat.replace('_', ' ')}: {value:.2f}" for stat, value in zip(stats, values)]
        # Combine into a single string with line breaks
        stats_text = '\n'.join(stat_strings)

        return stats_text
    else:
        return ""


# Define callback to update feature importance plot
@app.callback(
    Output('feature-importance-bar', 'figure'),
    Input('feature-dropdown', 'value')  # Dummy input to trigger callback
)
def update_feature_importance(_):
    # Use the feature importances calculated during model training
    global feature_importances
    if 'feature_importances' not in globals():
        # Re-train the Random Forest to get feature importances if not already done
        # Note: This is not ideal; better to compute feature_importances during initial training and store globally
        feature_columns = ['Picks', 'Bans', 'Presence', 'CSM', 'DPM', 'GPM', 'CSD@15', 'GD@15', 'XPD@15']
        if 'KDA_numeric' in preprocessed_data.columns:
            feature_columns.append('KDA_numeric')
        X = preprocessed_data[feature_columns]
        y = preprocessed_data['Winrate']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        rfr = RandomForestRegressor(n_estimators=100, random_state=42)
        rfr.fit(X_train_scaled, y_train)
        feature_importances = pd.Series(rfr.feature_importances_, index=X.columns).sort_values(ascending=False)

    fig = go.Figure([go.Bar(
        x=feature_importances.index,
        y=feature_importances.values,
        marker_color='lightgreen'
    )])
    fig.update_layout(title='Feature Importances from Random Forest Regressor',
                      xaxis_title='Features',
                      yaxis_title='Importance Score',
                      template=graph_template,
                      paper_bgcolor='#303030',
                      plot_bgcolor='#303030',
                      title_font=dict(color='#ffffff'),
                      font=dict(color='#ffffff'))
    return fig


# Define callback to update correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('feature-dropdown', 'value')  # Dummy input to trigger callback
)
def update_correlation_heatmap(_):
    corr = preprocessed_data[visualization_columns].corr()
    fig = px.imshow(corr,
                    text_auto=True,
                    color_continuous_scale='Viridis',
                    title='Correlation Heatmap of Features',
                    labels={'color': 'Correlation Coefficient'},
                    template=graph_template)
    fig.update_layout(paper_bgcolor='#303030', plot_bgcolor='#303030',
                      title_font=dict(color='#ffffff'),
                      font=dict(color='#ffffff'))
    return fig


# Define callback to update winrate histogram
@app.callback(
    Output('winrate-histogram', 'figure'),
    Input('feature-dropdown', 'value')  # Dummy input to trigger callback
)
def update_winrate_histogram(_):
    fig = px.histogram(preprocessed_data, x='Winrate',
                       nbins=20,
                       title='Distribution of Winrates',
                       labels={'Winrate': 'Winrate (%)'},
                       opacity=0.75,
                       template=graph_template)
    fig.update_layout(paper_bgcolor='#303030', plot_bgcolor='#303030',
                      title_font=dict(color='#ffffff'),
                      font=dict(color='#ffffff'))
    return fig


# Define callback to update metrics box plot
@app.callback(
    Output('metrics-box-plot', 'figure'),
    Input('feature-dropdown', 'value')  # Dummy input to trigger callback
)
def update_metrics_box_plot(_):
    fig = go.Figure()
    for stat in ['DPM', 'GPM', 'CSM', 'Winrate']:
        fig.add_trace(go.Box(y=preprocessed_data[stat], name=stat, marker_color='lightgreen'))

    fig.update_layout(title='Box Plots of Key Metrics',
                      yaxis_title='Values',
                      template=graph_template,
                      paper_bgcolor='#303030',
                      plot_bgcolor='#303030',
                      title_font=dict(color='#ffffff'),
                      font=dict(color='#ffffff'))
    return fig


# Define callback to update pair plot
@app.callback(
    Output('pair-plot', 'figure'),
    Input('feature-dropdown', 'value')  # Dummy input to trigger callback
)
def update_pair_plot(_):
    # Due to the limitations of Dash with Plotly's pairplot, we'll use scatter matrix instead
    fig = px.scatter_matrix(preprocessed_data,
                            dimensions=['DPM', 'GPM', 'CSM', 'Winrate'],
                            color='Winrate',
                            title='Pairwise Relationships of Key Metrics',
                            labels={'DPM': 'Damage Per Minute',
                                    'GPM': 'Gold Per Minute',
                                    'CSM': 'Creep Score per Minute',
                                    'Winrate': 'Winrate (%)'},
                            height=800,
                            template=graph_template,
                            color_continuous_scale='Bluered')
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(paper_bgcolor='#303030', plot_bgcolor='#303030',
                      title_font=dict(color='#ffffff'),
                      font=dict(color='#ffffff'))
    return fig


def main():
    # Preprocess the data and train models
    data = preprocess_data()

    # Machine Learning Models

    print("\nStarting machine learning models to predict 'Winrate'...")

    # Prepare the data for modeling
    # Selecting relevant features for prediction
    feature_columns = ['Picks', 'Bans', 'Presence', 'CSM', 'DPM', 'GPM', 'CSD@15', 'GD@15', 'XPD@15']

    # Include 'KDA_numeric' if it exists
    if 'KDA_numeric' in data.columns:
        feature_columns.append('KDA_numeric')

    X = data[feature_columns]
    y = data['Winrate']

    # Check for NaNs in X and y
    print("\nChecking for NaNs in features (X):")
    print(X.isnull().sum())
    print("\nChecking for NaNs in target (y):")
    print(y.isnull().sum())

    # Split the data into training and testing sets
    # Using 80% of the data for training and 20% for testing
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    # Scaling features to have zero mean and unit variance
    print("Applying feature scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Check for NaNs after scaling
    if np.isnan(X_train_scaled).any() or np.isnan(X_test_scaled).any():
        print("Error: NaNs detected in scaled features. Please check the preprocessing steps.")
        exit()

    # Linear Regression Model
    # Starting with a simple Linear Regression model
    print("Training Linear Regression model...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Predictions using Linear Regression
    print("Making predictions with Linear Regression model...")
    y_pred_lr = lr.predict(X_test_scaled)

    # Evaluation of Linear Regression
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print("\nLinear Regression Performance:")
    print(f"Mean Squared Error: {mse_lr:.2f}")
    print(f"R^2 Score: {r2_lr:.2f}")

    # Random Forest Regressor
    # Trying a Random Forest Regressor for comparison
    print("\nTraining Random Forest Regressor model...")
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(X_train_scaled, y_train)

    # Predictions using Random Forest Regressor
    print("Making predictions with Random Forest Regressor model...")
    y_pred_rfr = rfr.predict(X_test_scaled)

    # Evaluation of Random Forest Regressor
    mse_rfr = mean_squared_error(y_test, y_pred_rfr)
    r2_rfr = r2_score(y_test, y_pred_rfr)

    print("\nRandom Forest Regressor Performance:")
    print(f"Mean Squared Error: {mse_rfr:.2f}")
    print(f"R^2 Score: {r2_rfr:.2f}")

    # Feature Importance from Random Forest
    global feature_importances
    feature_importances = pd.Series(rfr.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances from Random Forest Regressor:")
    print(feature_importances)

    # Plotting Feature Importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

    # Compare the performance of both models
    print("\nComparison of Models:")
    print(f"Linear Regression MSE: {mse_lr:.2f}, R^2: {r2_lr:.2f}")
    print(f"Random Forest Regressor MSE: {mse_rfr:.2f}, R^2: {r2_rfr:.2f}")

    # Based on the R^2 scores, Random Forest Regressor may perform better


if __name__ == '__main__':
    main()
    print("\nLaunching interactive dashboard... Go to http://127.0.0.1:8050/ in your browser")
    app.run_server(debug=True)

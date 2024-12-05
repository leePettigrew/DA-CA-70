League of Legends Champions Analysis Dashboard

---

**Table of Contents**

- Overview
- Dataset
  - Dataset Overview
  - File Information
  - Provenance
- Features
- Installation
  - Prerequisites
  - Installing Dependencies
- Usage
  - Data Preprocessing and Model Training
  - Launching the Interactive Dashboard
- Project Structure
- Machine Learning Models
- Visualizations
- Acknowledgements

---

**Overview**

The League of Legends Champions Analysis Dashboard is a comprehensive tool designed to provide insights into champion performance, pick/ban rates, and various in-game metrics from the League of Legends Worlds 2024 - Swiss Stage. Leveraging data sourced from gol.gg (Games of Legends), this dashboard offers valuable information for analysts, coaches, and players to understand champion dynamics and make informed decisions.

---

**Dataset**

**Dataset Overview**

This dataset encapsulates detailed statistics from the League of Legends Worlds 2024 - Swiss Stage. It provides a granular view of champion performance, including pick and ban rates, winrates, and in-game metrics such as Damage per Minute (DPM), Gold per Minute (GPM), and more. The data is instrumental for understanding the strengths and weaknesses of each champion within the competitive landscape of the 2024 Worlds tournament.

**File Information**

- **Filename:** champions.csv

- **Columns:**

  - **Champion:** The name of the champion in the Swiss Stage.
  - **Picks:** The number of times each champion was picked during the Swiss Stage matches.
  - **Bans:** The number of times each champion was banned, indicating their perceived threat level.
  - **Presence:** The percentage of matches where the champion was either picked or banned.
  - **Wins:** The total number of games the champion won.
  - **Losses:** The total number of games the champion lost.
  - **Winrate:** The win percentage calculated from the champion's games.
  - **KDA:** Kills/Deaths/Assists: A performance metric representing how effectively the champion contributes to team fights.
  - **Avg BT:** Average Build Time: The average time required for the champion to complete their item build.
  - **GT:** Game Time: The length of time the champion was involved in matches, reported in HH:MM:SS format.
  - **CSM:** Creep Score per Minute: Average number of minions killed per minute, reflecting farming efficiency.
  - **DPM:** Damage per Minute: Average damage dealt by the champion per minute.
  - **GPM:** Gold per Minute: Gold earned by the champion per minute.
  - **CSD@15:** Creep Score Difference at 15 minutes: Difference in creep score compared to the opposing player by the 15-minute mark.
  - **GD@15:** Gold Difference at 15 minutes: The difference in gold between the champion and their opponent at the 15-minute mark.
  - **XPD@15:** Experience Difference at 15 minutes: The experience point difference between the champion and their opponent at the 15-minute mark.

**Provenance**

The dataset was collected from gol.gg (Games of Legends), an esports statistics platform. GOL provides comprehensive stats for competitive gaming but is not officially endorsed by Riot Games. Therefore, this data does not reflect the views or opinions of Riot Games or anyone involved in managing League of Legends.

---

**Features**

- **Data Preprocessing:** Cleans and transforms raw data for analysis and modeling.
- **Machine Learning Models:** Predicts champion winrates using Linear Regression and Random Forest Regressor.
- **Interactive Dashboard:** Visualizes champion statistics and dataset insights using Plotly and Dash.
- **Visualizations:** Includes scatter plots, bar charts, radar charts, box plots, heatmaps, and more for comprehensive data exploration.

---

**Installation**

**Prerequisites**

- **Python 3.7 or higher:** Ensure you have Python installed. You can download it from Python's official website.
- **pip:** Python package installer. It typically comes with Python. Verify by running `pip --version` in your command line.

**Installing Dependencies**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/leePettigrew/DA-CA-70.git
   cd DA-CA-70
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   ```

   - **Activate the Virtual Environment:**

     - **Windows:**

       ```bash
       venv\Scripts\activate
       ```

     - **macOS/Linux:**

       ```bash
       source venv/bin/activate
       ```

3. **Install Required Packages**

   Install all necessary Python packages using pip:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn plotly dash statsmodels
   ```



---

**Usage**

**Data Preprocessing and Model Training**

1. **Prepare the Dataset**

   Ensure that the `champions.csv` file is placed in the root directory of the project.

2. **Run the Preprocessing and Modeling Script**

   Execute the `data_preprocessing.py` script to preprocess the data, train machine learning models, and save the preprocessed data:

   ```bash
   python data_preprocessing.py
   ```

   - **What It Does:**
     - Loads and cleans the dataset.
     - Handles missing values and converts data types.
     - Performs feature engineering.
     - Trains Linear Regression and Random Forest Regressor models to predict champion winrates.
     - Evaluates model performance.
     - Saves the preprocessed data to `champions_preprocessed.csv`.

   - **Output:**
     - Console logs detailing each step and model performance metrics.
     - A `champions_preprocessed.csv` file containing the cleaned and engineered data.
     - Feature importance plot displayed using Matplotlib and Seaborn.

**Launching the Interactive Dashboard**

After preprocessing the data, you can launch the interactive dashboard to explore the insights.

1. **Ensure Preprocessed Data Exists**

   The `data_preprocessing.py` script saves the preprocessed data as `champions_preprocessed.csv`. Ensure this file is present in the project directory.

2. **Run the Dashboard**

   The `data_preprocessing.py` script also initializes and runs the Dash application. If not already running, execute:

   ```bash
   python data_preprocessing.py
   ```

   - **Access the Dashboard:**

     Open your web browser and navigate to [http://127.0.0.1:8050/](http://127.0.0.1:8050/) to interact with the dashboard.

   - **Dashboard Features:**
     - **Champion Selection:** Choose a champion to view detailed statistics.
     - **Scatter Plots:** Visualize relationships between winrate and selected features.
     - **Bar Charts & Radar Charts:** Compare champion-specific metrics.
     - **Correlation Heatmap:** Understand feature interdependencies.
     - **Histograms & Box Plots:** Analyze distributions of key metrics.
     - **Pair Plots:** Explore pairwise relationships between metrics.

---

**Project Structure**

```
DA-CA-70/
├── champions.csv
├── champions_preprocessed.csv
├── data_preprocessing.py
├── README.md
```

- **champions.csv:** Raw dataset containing champion statistics.
- **champions_preprocessed.csv:** Cleaned and processed data for analysis and modeling.
- **data_preprocessing.py:** Main script for data preprocessing, model training, and launching the dashboard.
- **README.md:** Project documentation.

---

**Machine Learning Models**

The project employs two machine learning models to predict champion winrates based on various in-game metrics:

1. **Linear Regression**
   - **Purpose:** Establish a baseline model for predicting winrate.
   - **Performance:**
     - **Mean Squared Error (MSE):** Indicates the average squared difference between predicted and actual winrates.
     - **R² Score:** Represents the proportion of variance in the dependent variable that's predictable from the independent variables.

2. **Random Forest Regressor**
   - **Purpose:** Improve prediction accuracy by capturing non-linear relationships and feature interactions.
   - **Performance:**
     - Typically exhibits higher R² scores and lower MSE compared to Linear Regression.
   - **Feature Importance:**
     - Identifies which features most significantly impact winrate predictions, aiding in strategic decision-making.

---

**Visualizations**

The dashboard offers a variety of interactive visualizations to facilitate data exploration:

- **Scatter Plots:** Display relationships between winrate and selected features with trendlines.
- **Bar Charts:** Show champion-specific statistics for easy comparison.
- **Radar Charts:** Visualize normalized metrics to assess overall champion performance.
- **Box Plots:** Compare selected champion's metrics against the entire dataset.
- **Correlation Heatmap:** Highlight correlations between different features.
- **Histograms:** Illustrate the distribution of winrates.
- **Pair Plots (Scatter Matrix):** Explore pairwise relationships between key metrics.

All visualizations are styled with a dark theme for enhanced readability and aesthetic appeal.

---

**Acknowledgements**

- **gol.gg (Games of Legends):** For providing comprehensive esports statistics.
- **Riot Games:** For creating League of Legends, the game analyzed in this project.
- **Open Source Libraries:** Thanks to the developers of pandas, NumPy, scikit-learn, Plotly, Dash, and other libraries that made this project possible.

---

**Disclaimer:** This project is not affiliated with Riot Games. All data is sourced from gol.gg and is intended for informational and analytical purposes only.

---

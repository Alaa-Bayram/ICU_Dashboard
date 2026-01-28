# ICU Cardiac Arrest Analytics Dashboard

A professional Streamlit dashboard for descriptive analysis of ICU cardiac arrest data.

## Features

- **Interactive Filters**: Filter data by gender, age, year, location, and ROSC status
- **Key Performance Indicators**: Real-time metrics for patient counts and survival rates
- **Demographics Analysis**: Gender distribution, age histograms, and temporal trends
- **Comorbidity Analysis**: Prevalence charts and survival comparisons by condition
- **Cardiac Arrest Characteristics**: Initial rhythm and event location distributions
- **Outcomes Analysis**: Survival cascade funnel and outcome comparisons
- **Data Explorer**: View and download filtered raw data

## Installation

### 1. Prerequisites

Make sure you have Python 3.8+ installed on your system.

### 2. Clone or Download

Download the project files to your desired directory.

### 3. Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add Your Data

Place your `ICU.xlsx` file in the same directory as `icu_dashboard.py`.

## Running the Dashboard

### From Terminal/Command Prompt

```bash
streamlit run icu_dashboard.py
```

### From PyCharm

1. Open the project in PyCharm
2. Configure Python interpreter with the virtual environment
3. Open the terminal in PyCharm
4. Run: `streamlit run icu_dashboard.py`

**Alternative PyCharm Run Configuration:**
1. Go to Run → Edit Configurations
2. Click + → Python
3. Set:
   - Script path: Path to your streamlit installation (e.g., `venv/lib/python3.x/site-packages/streamlit/__main__.py`)
   - Parameters: `run icu_dashboard.py`
   - Working directory: Your project folder

The dashboard will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
project_folder/
├── icu_dashboard.py      # Main dashboard application
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── ICU.xlsx             # Your data file (add this)
```

## Dashboard Sections

### 1. Key Performance Indicators
- Total patients (filtered count)
- ROSC rate
- 24-hour survival rate
- Survival to discharge rate
- Mean patient age

### 2. Demographics Analysis
- Gender distribution (pie chart)
- Age distribution histogram with mean line
- Age group breakdown
- Cardiac arrests by year trend

### 3. Comorbidities Analysis
- Comorbidity prevalence horizontal bar chart
- Survival rates comparison by comorbidity status

### 4. Cardiac Arrest Characteristics
- Initial cardiac rhythm distribution
- Event location breakdown

### 5. Outcomes Analysis
- Survival cascade funnel chart
- ROSC rate by initial cardiac rhythm
- Outcomes by gender comparison
- Survival to discharge by age group

### 6. Summary Statistics
- Numerical variable statistics
- Outcome summary table
- Comorbidity summary table

### 7. Data Explorer
- Interactive data table
- CSV download functionality

## Customization

### Modifying Colors

Edit the color schemes in the respective Plotly chart definitions:
- `color_discrete_sequence` for categorical colors
- `color_continuous_scale` for gradient colors

### Adding New Metrics

Add new metrics in the KPI section by creating additional `st.metric()` calls.

### Adding New Charts

Follow the existing pattern:
1. Calculate the data
2. Create a Plotly figure
3. Update layout for consistent styling
4. Display with `st.plotly_chart()`

## Troubleshooting

### "Data file not found" error
Ensure `ICU.xlsx` is in the same directory as the Python script.

### Streamlit not recognized
Make sure you've activated your virtual environment and installed requirements.

### Charts not displaying
Check that all required columns exist in your data file.

## Dependencies

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Plotly**: Interactive visualizations
- **OpenPyXL**: Excel file reading

## License

This project is for educational and analytical purposes.

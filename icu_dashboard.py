import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ICU Cardiac Arrest Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metric card styling */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    div[data-testid="metric-container"] > label {
        color: #495057;
        font-weight: 500;
    }
    
    div[data-testid="metric-container"] > div {
        color: #212529;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        color: white;
    }
    
    .dashboard-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .dashboard-header p {
        margin: 10px 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Section headers */
    .section-header {
        background-color: #f8f9fa;
        padding: 12px 20px;
        border-radius: 8px;
        border-left: 4px solid #1e3a5f;
        margin: 25px 0 15px 0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    /* Card container */
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        margin-bottom: 15px;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 14px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b6d4fe;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
@st.cache_data
def load_and_preprocess_data(file_path):
    """Load and clean the ICU dataset."""
    df = pd.read_excel(file_path)
    
    # Standardize categorical variables (handle case inconsistencies)
    yes_no_cols = ['CAD', 'Heart Failure', 'Heart Disease', 'Hypertension', 
                   'COPD', 'Diabetes', 'Cancer', 'Covid at Admission', 
                   'Smoking', 'ROSC (Y/N)', '24 Hours Survival', 'Survival to Discharge']
    
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace({'yes ': 'yes', 'no ': 'no'})
    
    # Standardize Gender
    df['Gender'] = df['Gender'].astype(str).str.upper().str.strip()
    df['Gender'] = df['Gender'].replace({'UNKNOWN': 'Unknown'})
    
    # Standardize Initial Cardiac Rhythm
    df['Initial Cardiac Rhythm'] = df['Initial Cardiac Rhythm'].astype(str).str.lower().str.strip()
    rhythm_mapping = {
        'asystol': 'Asystole', 'asystole': 'Asystole', 'asysol': 'Asystole', 'aystol': 'Asystole',
        'bradycardia': 'Bradycardia', 'sinus bradycardia': 'Bradycardia',
        'vf': 'VF/VT', 'v-fib': 'VF/VT', 'v tach': 'VF/VT', 'v-tach': 'VF/VT',
        'pea': 'PEA',
        'af': 'AF/FA', 'fa': 'AF/FA',
        'sinus': 'Sinus', 'sr': 'Sinus',
        'unknown': 'Unknown', 'no': 'Unknown', 'no cpr': 'Unknown', '1x 150j': 'Unknown'
    }
    df['Initial Cardiac Rhythm'] = df['Initial Cardiac Rhythm'].replace(rhythm_mapping)
    
    # Convert Age to numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    
    # Create age groups
    bins = [0, 30, 45, 60, 75, 150]
    labels = ['<30', '30-44', '45-59', '60-74', '75+']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    return df


def calculate_survival_rate(df, outcome_col, filter_col=None, filter_val=None):
    """Calculate survival rate with optional filtering."""
    if filter_col and filter_val:
        subset = df[df[filter_col] == filter_val]
    else:
        subset = df
    
    valid = subset[subset[outcome_col].isin(['yes', 'no'])]
    if len(valid) == 0:
        return 0
    return (valid[outcome_col] == 'yes').sum() / len(valid) * 100


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Dashboard Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>🏥 ICU Cardiac Arrest Analytics Dashboard</h1>
        <p>Comprehensive descriptive analysis of in-hospital cardiac arrest outcomes</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_and_preprocess_data("ICU.xlsx")
    except FileNotFoundError:
        st.error("⚠️ Data file 'ICU.xlsx' not found. Please ensure the file is in the same directory as this script.")
        st.stop()
    
    # ========================================================================
    # SIDEBAR - FILTERS
    # ========================================================================
    with st.sidebar:
        st.markdown("## 🔍 Data Filters")
        st.markdown("---")
        
        # Gender filter
        gender_options = ['All'] + sorted(df['Gender'].unique().tolist())
        selected_gender = st.selectbox("Gender", gender_options)
        
        # Age range filter
        age_min = int(df['Age'].min()) if pd.notna(df['Age'].min()) else 0
        age_max = int(df['Age'].max()) if pd.notna(df['Age'].max()) else 100
        age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))
        
        # Year filter
        year_options = ['All'] + sorted(df['Arrest Year'].unique().tolist())
        selected_year = st.selectbox("Arrest Year", year_options)
        
        # Event Location filter
        location_options = ['All'] + sorted(df['Event Location'].unique().tolist())
        selected_location = st.selectbox("Event Location", location_options)
        
        # ROSC filter
        rosc_options = ['All', 'yes', 'no']
        selected_rosc = st.selectbox("ROSC Status", rosc_options)
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.info(f"**Total Records:** {len(df):,}")
        
    # Apply filters
    filtered_df = df.copy()
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
    if selected_year != 'All':
        filtered_df = filtered_df[filtered_df['Arrest Year'] == selected_year]
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['Event Location'] == selected_location]
    if selected_rosc != 'All':
        filtered_df = filtered_df[filtered_df['ROSC (Y/N)'] == selected_rosc]
    
    # ========================================================================
    # KEY PERFORMANCE INDICATORS
    # ========================================================================
    st.markdown('<div class="section-header"><h3>📈 Key Performance Indicators</h3></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df)/len(df)*100:.1f}% of total" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        rosc_rate = calculate_survival_rate(filtered_df, 'ROSC (Y/N)')
        st.metric(
            label="ROSC Rate",
            value=f"{rosc_rate:.1f}%",
            help="Return of Spontaneous Circulation"
        )
    
    with col3:
        survival_24h = calculate_survival_rate(filtered_df, '24 Hours Survival')
        st.metric(
            label="24-Hour Survival",
            value=f"{survival_24h:.1f}%"
        )
    
    with col4:
        survival_discharge = calculate_survival_rate(filtered_df, 'Survival to Discharge')
        st.metric(
            label="Survival to Discharge",
            value=f"{survival_discharge:.1f}%"
        )
    
    with col5:
        avg_age = filtered_df['Age'].mean()
        st.metric(
            label="Mean Age",
            value=f"{avg_age:.1f} yrs" if pd.notna(avg_age) else "N/A"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================================================================
    # DEMOGRAPHICS ANALYSIS
    # ========================================================================
    st.markdown('<div class="section-header"><h3>👥 Demographics Analysis</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender Distribution
        gender_counts = filtered_df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        fig_gender = px.pie(
            gender_counts, 
            values='Count', 
            names='Gender',
            title='<b>Gender Distribution</b>',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig_gender.update_traces(textposition='inside', textinfo='percent+label')
        fig_gender.update_layout(
            font=dict(family="Arial", size=12),
            title_font_size=16,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Age Distribution
        fig_age = px.histogram(
            filtered_df.dropna(subset=['Age']), 
            x='Age', 
            nbins=20,
            title='<b>Age Distribution</b>',
            color_discrete_sequence=['#1e3a5f']
        )
        fig_age.update_layout(
            xaxis_title="Age (years)",
            yaxis_title="Number of Patients",
            font=dict(family="Arial", size=12),
            title_font_size=16,
            bargap=0.1
        )
        fig_age.add_vline(x=filtered_df['Age'].mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {filtered_df['Age'].mean():.1f}")
        st.plotly_chart(fig_age, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Group Distribution
        age_group_counts = filtered_df['Age Group'].value_counts().sort_index().reset_index()
        age_group_counts.columns = ['Age Group', 'Count']
        fig_age_group = px.bar(
            age_group_counts,
            x='Age Group',
            y='Count',
            title='<b>Patients by Age Group</b>',
            color='Count',
            color_continuous_scale='Blues'
        )
        fig_age_group.update_layout(
            xaxis_title="Age Group",
            yaxis_title="Number of Patients",
            font=dict(family="Arial", size=12),
            title_font_size=16,
            showlegend=False
        )
        st.plotly_chart(fig_age_group, use_container_width=True)
    
    with col2:
        # Arrests by Year
        year_counts = filtered_df['Arrest Year'].value_counts().sort_index().reset_index()
        year_counts.columns = ['Year', 'Count']
        fig_year = px.line(
            year_counts,
            x='Year',
            y='Count',
            title='<b>Cardiac Arrests by Year</b>',
            markers=True
        )
        fig_year.update_traces(line_color='#1e3a5f', marker_size=10)
        fig_year.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Arrests",
            font=dict(family="Arial", size=12),
            title_font_size=16
        )
        st.plotly_chart(fig_year, use_container_width=True)
    
    # ========================================================================
    # COMORBIDITIES ANALYSIS
    # ========================================================================
    st.markdown('<div class="section-header"><h3>🩺 Comorbidities Analysis</h3></div>', unsafe_allow_html=True)
    
    comorbidities = ['Heart Disease', 'Hypertension', 'Heart Failure', 'Diabetes', 
                     'Cancer', 'COPD', 'CAD', 'Covid at Admission', 'Smoking']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Comorbidity Prevalence
        comorb_data = []
        for comorb in comorbidities:
            yes_count = (filtered_df[comorb] == 'yes').sum()
            total = filtered_df[comorb].isin(['yes', 'no']).sum()
            prevalence = (yes_count / total * 100) if total > 0 else 0
            comorb_data.append({'Comorbidity': comorb, 'Prevalence (%)': prevalence, 'Count': yes_count})
        
        comorb_df = pd.DataFrame(comorb_data).sort_values('Prevalence (%)', ascending=True)
        
        fig_comorb = px.bar(
            comorb_df,
            y='Comorbidity',
            x='Prevalence (%)',
            orientation='h',
            title='<b>Comorbidity Prevalence</b>',
            color='Prevalence (%)',
            color_continuous_scale='Reds',
            text='Prevalence (%)'
        )
        fig_comorb.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_comorb.update_layout(
            xaxis_title="Prevalence (%)",
            yaxis_title="",
            font=dict(family="Arial", size=12),
            title_font_size=16,
            showlegend=False
        )
        st.plotly_chart(fig_comorb, use_container_width=True)
    
    with col2:
        # Comorbidity vs Survival to Discharge
        survival_by_comorb = []
        for comorb in comorbidities:
            for status in ['yes', 'no']:
                subset = filtered_df[(filtered_df[comorb] == status) & 
                                    filtered_df['Survival to Discharge'].isin(['yes', 'no'])]
                if len(subset) > 0:
                    surv_rate = (subset['Survival to Discharge'] == 'yes').sum() / len(subset) * 100
                    survival_by_comorb.append({
                        'Comorbidity': comorb,
                        'Status': f'With {comorb}' if status == 'yes' else f'Without {comorb}',
                        'Has Condition': 'Yes' if status == 'yes' else 'No',
                        'Survival Rate (%)': surv_rate
                    })
        
        surv_comorb_df = pd.DataFrame(survival_by_comorb)
        
        fig_surv_comorb = px.bar(
            surv_comorb_df,
            x='Comorbidity',
            y='Survival Rate (%)',
            color='Has Condition',
            barmode='group',
            title='<b>Survival to Discharge by Comorbidity Status</b>',
            color_discrete_map={'Yes': '#c0392b', 'No': '#27ae60'}
        )
        fig_surv_comorb.update_layout(
            xaxis_title="",
            yaxis_title="Survival Rate (%)",
            font=dict(family="Arial", size=12),
            title_font_size=16,
            xaxis_tickangle=-45,
            legend_title="Has Condition"
        )
        st.plotly_chart(fig_surv_comorb, use_container_width=True)
    
    # ========================================================================
    # CARDIAC ARREST CHARACTERISTICS
    # ========================================================================
    st.markdown('<div class="section-header"><h3>❤️ Cardiac Arrest Characteristics</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Initial Cardiac Rhythm
        rhythm_counts = filtered_df['Initial Cardiac Rhythm'].value_counts().reset_index()
        rhythm_counts.columns = ['Rhythm', 'Count']
        rhythm_counts = rhythm_counts[rhythm_counts['Rhythm'] != 'Unknown']
        
        fig_rhythm = px.pie(
            rhythm_counts,
            values='Count',
            names='Rhythm',
            title='<b>Initial Cardiac Rhythm Distribution</b>',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_rhythm.update_traces(textposition='inside', textinfo='percent+label')
        fig_rhythm.update_layout(
            font=dict(family="Arial", size=12),
            title_font_size=16
        )
        st.plotly_chart(fig_rhythm, use_container_width=True)
    
    with col2:
        # Event Location
        location_counts = filtered_df['Event Location'].value_counts().reset_index()
        location_counts.columns = ['Location', 'Count']
        
        fig_location = px.bar(
            location_counts,
            x='Location',
            y='Count',
            title='<b>Event Location Distribution</b>',
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig_location.update_layout(
            xaxis_title="Location",
            yaxis_title="Number of Events",
            font=dict(family="Arial", size=12),
            title_font_size=16,
            showlegend=False
        )
        st.plotly_chart(fig_location, use_container_width=True)
    
    # ========================================================================
    # OUTCOMES ANALYSIS
    # ========================================================================
    st.markdown('<div class="section-header"><h3>📊 Outcomes Analysis</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Survival Cascade (Funnel Chart)
        total = len(filtered_df)
        rosc_yes = (filtered_df['ROSC (Y/N)'] == 'yes').sum()
        surv_24h = (filtered_df['24 Hours Survival'] == 'yes').sum()
        surv_discharge = (filtered_df['Survival to Discharge'] == 'yes').sum()
        
        fig_funnel = go.Figure(go.Funnel(
            y=['Total Arrests', 'ROSC Achieved', '24-Hour Survival', 'Survival to Discharge'],
            x=[total, rosc_yes, surv_24h, surv_discharge],
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=['#1e3a5f', '#2d5a87', '#3d7ab0', '#4d9ad9'])
        ))
        fig_funnel.update_layout(
            title='<b>Survival Cascade</b>',
            font=dict(family="Arial", size=12),
            title_font_size=16
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col2:
        # ROSC by Initial Rhythm
        rosc_by_rhythm = []
        for rhythm in filtered_df['Initial Cardiac Rhythm'].unique():
            if rhythm != 'Unknown':
                subset = filtered_df[(filtered_df['Initial Cardiac Rhythm'] == rhythm) &
                                    filtered_df['ROSC (Y/N)'].isin(['yes', 'no'])]
                if len(subset) > 10:  # Only include rhythms with sufficient data
                    rosc_rate = (subset['ROSC (Y/N)'] == 'yes').sum() / len(subset) * 100
                    rosc_by_rhythm.append({'Rhythm': rhythm, 'ROSC Rate (%)': rosc_rate, 'N': len(subset)})
        
        if rosc_by_rhythm:
            rosc_rhythm_df = pd.DataFrame(rosc_by_rhythm).sort_values('ROSC Rate (%)', ascending=False)
            
            fig_rosc_rhythm = px.bar(
                rosc_rhythm_df,
                x='Rhythm',
                y='ROSC Rate (%)',
                title='<b>ROSC Rate by Initial Cardiac Rhythm</b>',
                color='ROSC Rate (%)',
                color_continuous_scale='RdYlGn',
                text='ROSC Rate (%)'
            )
            fig_rosc_rhythm.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_rosc_rhythm.update_layout(
                xaxis_title="Initial Rhythm",
                yaxis_title="ROSC Rate (%)",
                font=dict(family="Arial", size=12),
                title_font_size=16,
                showlegend=False
            )
            st.plotly_chart(fig_rosc_rhythm, use_container_width=True)
    
    # ========================================================================
    # SURVIVAL BY DEMOGRAPHICS
    # ========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        # Survival by Gender
        surv_gender = []
        for gender in ['M', 'F']:
            subset = filtered_df[(filtered_df['Gender'] == gender) &
                                filtered_df['Survival to Discharge'].isin(['yes', 'no'])]
            if len(subset) > 0:
                rosc = (subset['ROSC (Y/N)'] == 'yes').sum() / len(subset) * 100
                s24 = (subset['24 Hours Survival'] == 'yes').sum() / len(subset) * 100
                sd = (subset['Survival to Discharge'] == 'yes').sum() / len(subset) * 100
                surv_gender.append({'Gender': gender, 'ROSC': rosc, '24-Hour Survival': s24, 'Discharge': sd})
        
        if surv_gender:
            surv_gender_df = pd.DataFrame(surv_gender)
            surv_gender_long = surv_gender_df.melt(id_vars=['Gender'], var_name='Outcome', value_name='Rate (%)')
            
            fig_surv_gender = px.bar(
                surv_gender_long,
                x='Outcome',
                y='Rate (%)',
                color='Gender',
                barmode='group',
                title='<b>Outcomes by Gender</b>',
                color_discrete_map={'M': '#3498db', 'F': '#e74c3c'}
            )
            fig_surv_gender.update_layout(
                xaxis_title="",
                yaxis_title="Rate (%)",
                font=dict(family="Arial", size=12),
                title_font_size=16
            )
            st.plotly_chart(fig_surv_gender, use_container_width=True)
    
    with col2:
        # Survival by Age Group
        surv_age = []
        for age_group in filtered_df['Age Group'].dropna().unique():
            subset = filtered_df[(filtered_df['Age Group'] == age_group) &
                                filtered_df['Survival to Discharge'].isin(['yes', 'no'])]
            if len(subset) > 5:
                sd = (subset['Survival to Discharge'] == 'yes').sum() / len(subset) * 100
                surv_age.append({'Age Group': str(age_group), 'Survival Rate (%)': sd, 'N': len(subset)})
        
        if surv_age:
            surv_age_df = pd.DataFrame(surv_age)
            # Sort by age group order
            age_order = ['<30', '30-44', '45-59', '60-74', '75+']
            surv_age_df['Age Group'] = pd.Categorical(surv_age_df['Age Group'], categories=age_order, ordered=True)
            surv_age_df = surv_age_df.sort_values('Age Group')
            
            fig_surv_age = px.bar(
                surv_age_df,
                x='Age Group',
                y='Survival Rate (%)',
                title='<b>Survival to Discharge by Age Group</b>',
                color='Survival Rate (%)',
                color_continuous_scale='RdYlGn',
                text='Survival Rate (%)'
            )
            fig_surv_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_surv_age.update_layout(
                xaxis_title="Age Group",
                yaxis_title="Survival Rate (%)",
                font=dict(family="Arial", size=12),
                title_font_size=16,
                showlegend=False
            )
            st.plotly_chart(fig_surv_age, use_container_width=True)
    
    # ========================================================================
    # DETAILED STATISTICS TABLE
    # ========================================================================
    st.markdown('<div class="section-header"><h3>📋 Summary Statistics</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Numerical Variables")
        numeric_stats = filtered_df[['Age']].describe().round(2)
        numeric_stats.columns = ['Age (years)']
        st.dataframe(numeric_stats, use_container_width=True)
        
        st.markdown("#### Outcome Summary")
        outcomes_summary = pd.DataFrame({
            'Metric': ['Total Patients', 'ROSC Achieved', '24-Hour Survival', 'Survival to Discharge'],
            'Count': [
                len(filtered_df),
                (filtered_df['ROSC (Y/N)'] == 'yes').sum(),
                (filtered_df['24 Hours Survival'] == 'yes').sum(),
                (filtered_df['Survival to Discharge'] == 'yes').sum()
            ],
            'Rate (%)': [
                100.0,
                calculate_survival_rate(filtered_df, 'ROSC (Y/N)'),
                calculate_survival_rate(filtered_df, '24 Hours Survival'),
                calculate_survival_rate(filtered_df, 'Survival to Discharge')
            ]
        })
        outcomes_summary['Rate (%)'] = outcomes_summary['Rate (%)'].round(1)
        st.dataframe(outcomes_summary.set_index('Metric'), use_container_width=True)
    
    with col2:
        st.markdown("#### Comorbidity Summary")
        comorb_summary = pd.DataFrame(comorb_data)
        comorb_summary = comorb_summary.sort_values('Prevalence (%)', ascending=False)
        comorb_summary['Prevalence (%)'] = comorb_summary['Prevalence (%)'].round(1)
        comorb_summary = comorb_summary.rename(columns={'Count': 'Patients with Condition'})
        st.dataframe(comorb_summary.set_index('Comorbidity'), use_container_width=True)
    
    # ========================================================================
    # RAW DATA VIEWER
    # ========================================================================
    st.markdown('<div class="section-header"><h3>📄 Data Explorer</h3></div>', unsafe_allow_html=True)
    
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_icu_data.csv",
            mime="text/csv"
        )
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 20px;'>
        <p><strong>ICU Cardiac Arrest Analytics Dashboard</strong></p>
        <p>Data contains {total} patient records | Filtered view: {filtered} records</p>
    </div>
    """.format(total=len(df), filtered=len(filtered_df)), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

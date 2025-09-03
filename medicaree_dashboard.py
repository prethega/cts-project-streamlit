import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medicare Risk Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .risk-tier-1 { background: linear-gradient(135deg, #10b981 0%, #34d399 100%); }
    .risk-tier-2 { background: linear-gradient(135deg, #06b6d4 0%, #67e8f9 100%); }
    .risk-tier-3 { background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); }
    .risk-tier-4 { background: linear-gradient(135deg, #ef4444 0%, #f87171 100%); }
    .risk-tier-5 { background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%); }
    .department-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MockMedicareDashboard:
    """Mock Medicare Dashboard with realistic synthetic data"""
    
    def __init__(self):
        self.initialize_synthetic_data()
        self.setup_models()
    
    def initialize_synthetic_data(self):
        """Generate 10,000 synthetic patients with realistic distributions"""
        np.random.seed(42)
        n_patients = 10000
        
        # Generate patient IDs
        patient_ids = [f"PAT{str(i).zfill(6)}" for i in range(1, n_patients + 1)]
        
        # Realistic age distribution for Medicare (65+)
        ages = np.random.choice(
            range(65, 95), 
            n_patients, 
            p=self.generate_age_distribution()
        )
        
        # Gender distribution (slightly more female in Medicare)
        gender_male = np.random.choice([0, 1], n_patients, p=[0.56, 0.44])
        
        # Race distribution based on Medicare demographics
        race_probs = [0.78, 0.09, 0.13]  # White, Black, Other
        race_choices = np.random.choice([0, 1, 2], n_patients, p=race_probs)
        race_white = (race_choices == 0).astype(int)
        race_black = (race_choices == 1).astype(int)
        race_other = (race_choices == 2).astype(int)
        
        # Age categories
        age_65_74 = ((ages >= 65) & (ages < 75)).astype(int)
        age_75_84 = ((ages >= 75) & (ages < 85)).astype(int)
        age_85_plus = (ages >= 85).astype(int)
        
        # Chronic conditions with realistic prevalence and correlation
        chronic_conditions = self.generate_chronic_conditions(ages, n_patients)
        
        # Calculate condition counts
        condition_cols = ['SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR',
                          'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT',
                          'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA']
        
        chronic_condition_count = np.sum([chronic_conditions[col] for col in condition_cols], axis=0)
        
        # High-impact conditions
        high_impact_cols = ['SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD']
        high_impact_conditions = np.sum([chronic_conditions[col] for col in high_impact_cols], axis=0)
        
        # Healthcare utilization based on age and conditions
        utilization = self.generate_utilization_data(ages, chronic_condition_count, n_patients)
        
        # Create the main dataframe
        self.df = pd.DataFrame({
            'DESYNPUF_ID': patient_ids,
            'age': ages,
            'gender_male': gender_male,
            'race_white': race_white,
            'race_black': race_black,
            'race_other': race_other,
            'age_65_74': age_65_74,
            'age_75_84': age_75_84,
            'age_85_plus': age_85_plus,
            'chronic_condition_count': chronic_condition_count,
            'high_impact_conditions': high_impact_conditions,
            **chronic_conditions,
            **utilization
        })
        
        # Generate risk scores and tiers
        self.generate_risk_scores_and_tiers()
        self.assign_departments()
        self.calculate_roi_metrics()
    
    def generate_age_distribution(self):
        """Generate realistic Medicare age distribution"""
        # Based on actual Medicare demographics
        age_ranges = range(65, 95)
        # Peak around 70-75, declining after 85
        probs = []
        for age in age_ranges:
            if age < 70:
                prob = 0.8 + (age - 65) * 0.04
            elif age < 80:
                prob = 1.0 - (age - 70) * 0.02
            elif age < 90:
                prob = 0.8 - (age - 80) * 0.06
            else:
                prob = 0.2 - (age - 90) * 0.03
            probs.append(max(prob, 0.1))
        
        probs = np.array(probs)
        return probs / probs.sum()
    
    def generate_chronic_conditions(self, ages, n_patients):
        """Generate realistic chronic condition distributions"""
        conditions = {}
        
        # Age-dependent condition probabilities
        base_probs = {
            'SP_DIABETES': 0.28,    # Diabetes: 28% in Medicare
            'SP_CHF': 0.14,         # Heart Failure: 14%
            'SP_ISCHMCHT': 0.25,    # Ischemic Heart: 25%
            'SP_COPD': 0.11,        # COPD: 11%
            'SP_DEPRESSN': 0.18,    # Depression: 18%
            'SP_CHRNKIDN': 0.09,    # Kidney Disease: 9%
            'SP_CNCR': 0.07,        # Cancer: 7%
            'SP_OSTEOPRS': 0.15,    # Osteoporosis: 15%
            'SP_RA_OA': 0.23,       # Arthritis: 23%
            'SP_STRKETIA': 0.06,    # Stroke: 6%
            'SP_ALZHDMTA': 0.04     # Alzheimer's: 4%
        }
        
        for condition, base_prob in base_probs.items():
            # Age adjustment
            age_factor = 1 + (ages - 65) * 0.015  # Increases with age
            condition_probs = np.clip(base_prob * age_factor, 0.01, 0.8)
            conditions[condition] = np.random.binomial(1, condition_probs)
        
        return conditions
    
    def generate_utilization_data(self, ages, condition_counts, n_patients):
        """Generate realistic healthcare utilization"""
        # Base utilization rates increase with age and conditions
        age_factor = 1 + (ages - 65) * 0.02
        condition_factor = 1 + condition_counts * 0.3
        
        # Inpatient admissions (Poisson distribution)
        ip_rates = np.clip(0.15 * age_factor * condition_factor, 0, 3)
        inpatient_admissions = np.random.poisson(ip_rates)
        
        # Inpatient days
        inpatient_days = inpatient_admissions * np.random.poisson(4, n_patients)
        
        # Outpatient visits
        op_rates = np.clip(8 * age_factor * condition_factor, 2, 50)
        outpatient_visits = np.random.poisson(op_rates)
        
        # Derived indicators
        prior_hospitalization = (inpatient_admissions > 0).astype(int)
        frequent_ed_user = (outpatient_visits > np.percentile(outpatient_visits, 90)).astype(int)
        
        # Costs based on utilization
        inpatient_costs = inpatient_admissions * np.random.normal(12000, 3000, n_patients)
        inpatient_costs = np.clip(inpatient_costs, 0, None)
        
        outpatient_costs = outpatient_visits * np.random.normal(350, 100, n_patients)
        outpatient_costs = np.clip(outpatient_costs, 0, None)
        
        carrier_costs = np.random.normal(2000, 800, n_patients)
        carrier_costs = np.clip(carrier_costs, 0, None)
        
        total_medicare_costs = inpatient_costs + outpatient_costs + carrier_costs
        high_cost_patient = (total_medicare_costs > np.percentile(total_medicare_costs, 85)).astype(int)
        
        return {
            'inpatient_admissions': inpatient_admissions,
            'inpatient_days': inpatient_days,
            'outpatient_visits': outpatient_visits,
            'prior_hospitalization': prior_hospitalization,
            'frequent_ed_user': frequent_ed_user,
            'total_medicare_costs': total_medicare_costs,
            'high_cost_patient': high_cost_patient,
            'MEDREIMB_IP': inpatient_costs,
            'MEDREIMB_OP': outpatient_costs,  # Fixed: was medreimb_op
            'MEDREIMB_CAR': carrier_costs
        }
    
    def generate_risk_scores_and_tiers(self):
        """Generate risk scores and assign tiers"""
        # Calculate composite risk score
        age_risk = (self.df['age'] - 65) / 30  # Normalize age
        condition_risk = self.df['chronic_condition_count'] / 10
        utilization_risk = np.clip(self.df['inpatient_admissions'] / 5, 0, 1)
        cost_risk = np.clip(self.df['total_medicare_costs'] / 50000, 0, 1)
        
        composite_risk = (
            age_risk * 0.2 + 
            condition_risk * 0.4 + 
            utilization_risk * 0.25 + 
            cost_risk * 0.15
        )
        
        # Add some randomness for realism
        composite_risk += np.random.normal(0, 0.1, len(self.df))
        composite_risk = np.clip(composite_risk, 0, 1)
        
        # Assign tiers based on percentiles (realistic distribution)
        self.df['risk_tier'] = 1
        self.df.loc[composite_risk >= np.percentile(composite_risk, 50), 'risk_tier'] = 2
        self.df.loc[composite_risk >= np.percentile(composite_risk, 75), 'risk_tier'] = 3
        self.df.loc[composite_risk >= np.percentile(composite_risk, 93), 'risk_tier'] = 4
        self.df.loc[composite_risk >= np.percentile(composite_risk, 97), 'risk_tier'] = 5
        
        # Risk tier labels and interventions with adjusted costs for positive ROI
        tier_mapping = {
            1: {'label': 'Low Risk', 'intervention': 'Preventive Care', 'frequency': 'Annual', 'cost': 150},
            2: {'label': 'Low-Moderate Risk', 'intervention': 'Enhanced Wellness', 'frequency': 'Semi-Annual', 'cost': 200},
            3: {'label': 'Moderate Risk', 'intervention': 'Care Coordination', 'frequency': 'Quarterly', 'cost': 300},
            4: {'label': 'High Risk', 'intervention': 'Case Management', 'frequency': 'Monthly', 'cost': 400},
            5: {'label': 'Critical Risk', 'intervention': 'Intensive Management', 'frequency': 'Weekly', 'cost': 500}
        }
        
        for tier, info in tier_mapping.items():
            mask = self.df['risk_tier'] == tier
            self.df.loc[mask, 'risk_tier_label'] = info['label']
            self.df.loc[mask, 'care_intervention'] = info['intervention']
            self.df.loc[mask, 'care_frequency'] = info['frequency']
            self.df.loc[mask, 'annual_intervention_cost'] = info['cost']
        
        # Generate model-specific risk scores
        self.df['risk_30d_hospitalization'] = np.clip(composite_risk * 0.6 + np.random.normal(0, 0.1, len(self.df)), 0.01, 0.8)
        self.df['risk_60d_hospitalization'] = np.clip(composite_risk * 0.7 + np.random.normal(0, 0.12, len(self.df)), 0.02, 0.85)
        self.df['risk_90d_hospitalization'] = np.clip(composite_risk * 0.8 + np.random.normal(0, 0.15, len(self.df)), 0.03, 0.9)
        self.df['mortality_risk'] = np.clip(composite_risk * 0.3 + np.random.normal(0, 0.08, len(self.df)), 0.001, 0.4)
    
    def assign_departments(self):
        """Assign patients to relevant departments based on conditions"""
        departments = []
        for _, row in self.df.iterrows():
            dept_list = []
            
            # Cardiology
            if row['SP_CHF'] == 1 or row['SP_ISCHMCHT'] == 1:
                dept_list.append('Cardiology')
            
            # Endocrinology
            if row['SP_DIABETES'] == 1:
                dept_list.append('Endocrinology')
            
            # Pulmonology
            if row['SP_COPD'] == 1:
                dept_list.append('Pulmonology')
            
            # Oncology
            if row['SP_CNCR'] == 1:
                dept_list.append('Oncology')
            
            # Nephrology
            if row['SP_CHRNKIDN'] == 1:
                dept_list.append('Nephrology')
            
            # Neurology
            if row['SP_STRKETIA'] == 1 or row['SP_ALZHDMTA'] == 1:
                dept_list.append('Neurology')
            
            # Psychiatry
            if row['SP_DEPRESSN'] == 1:
                dept_list.append('Psychiatry')
            
            # Rheumatology
            if row['SP_RA_OA'] == 1 or row['SP_OSTEOPRS'] == 1:
                dept_list.append('Rheumatology')
            
            # General Medicine for those with no specific conditions or primary care
            if not dept_list or row['age'] >= 80:
                dept_list.append('General Medicine')
            
            departments.append('; '.join(dept_list))
        
        self.df['primary_departments'] = departments
    
    def calculate_roi_metrics(self):
        """Calculate ROI metrics for each patient with improved savings calculations"""
        # Higher prevention rates for better ROI
        prevention_rates = {1: 0.05, 2: 0.12, 3: 0.25, 4: 0.40, 5: 0.55}
        self.df['prevention_rate'] = self.df['risk_tier'].map(prevention_rates)
        
        # Enhanced cost savings calculation with multiple benefit streams
        avg_hospitalization_cost = 15000  # Increased from 12000
        avg_readmission_cost = 10000      # Increased from 8000
        avg_emergency_cost = 2500         # New: Emergency visit prevention
        
        # Base preventable cost per episode
        base_preventable_cost = (avg_hospitalization_cost + avg_readmission_cost + avg_emergency_cost) / 2
        
        # Risk-adjusted preventable episodes
        self.df['prevented_hospitalizations'] = self.df['risk_30d_hospitalization'] * self.df['prevention_rate']
        self.df['prevented_readmissions'] = self.df['risk_60d_hospitalization'] * self.df['prevention_rate'] * 0.8
        self.df['prevented_emergency_visits'] = self.df['risk_30d_hospitalization'] * self.df['prevention_rate'] * 2.5
        
        # Multiple savings streams
        self.df['hospitalization_savings'] = self.df['prevented_hospitalizations'] * avg_hospitalization_cost
        self.df['readmission_savings'] = self.df['prevented_readmissions'] * avg_readmission_cost
        self.df['emergency_savings'] = self.df['prevented_emergency_visits'] * avg_emergency_cost
        
        # Quality bonus based on improved outcomes
        quality_multiplier = self.df['risk_tier'].map({1: 1.1, 2: 1.15, 3: 1.2, 4: 1.3, 5: 1.4})
        
        # Total cost savings
        direct_savings = (self.df['hospitalization_savings'] + 
                         self.df['readmission_savings'] + 
                         self.df['emergency_savings'])
        
        self.df['cost_savings'] = direct_savings * quality_multiplier
        
        # Net savings and ROI
        self.df['net_savings'] = self.df['cost_savings'] - self.df['annual_intervention_cost']
        self.df['roi_percentage'] = (self.df['net_savings'] / self.df['annual_intervention_cost']) * 100
    
    def setup_models(self):
        """Setup mock models for prediction"""
        # Feature columns matching your original code
        self.feature_columns = [
            'age', 'gender_male', 'race_white', 'race_black', 'race_other',
            'age_65_74', 'age_75_84', 'age_85_plus',
            'SP_CHF', 'SP_DIABETES', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD',
            'SP_DEPRESSN', 'SP_ISCHMCHT', 'SP_STRKETIA', 'SP_ALZHDMTA',
            'SP_OSTEOPRS', 'SP_RA_OA',
            'chronic_condition_count', 'high_impact_conditions',
            'inpatient_admissions', 'inpatient_days', 'outpatient_visits',
            'prior_hospitalization', 'frequent_ed_user',
            'total_medicare_costs', 'high_cost_patient',
            'MEDREIMB_IP', 'MEDREIMB_OP', 'MEDREIMB_CAR'
        ]
        
        # Mock preprocessing pipeline
        self.preprocessing_pipeline = StandardScaler()
        X = self.df[self.feature_columns]
        self.preprocessing_pipeline.fit(X)

# Initialize the dashboard
@st.cache_data
def load_dashboard_data():
    return MockMedicareDashboard()

# Load data
dashboard = load_dashboard_data()
df = dashboard.df

# Main header
st.markdown("""
<div class="main-header">
    <h3>🏥 Medcare Risk Prediction & Management Dashboard</h3>
    <p>Advanced AI-Powered Risk Stratification and Care Optimization System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://static.vecteezy.com/system/resources/previews/023/740/386/non_2x/medicine-doctor-with-stethoscope-in-hand-on-hospital-background-medical-technology-healthcare-and-medical-concept-photo.jpg",
                  caption="Healthcare Analytics Dashboard")

st.sidebar.markdown("### 🎛️ Dashboard Controls")
page = st.sidebar.selectbox(
    "Select Dashboard View",
    ["📊 Overview & Analytics", "🔍 Patient Search", "➕ New Patient Prediction", "🏥 Department Management", "💰 ROI Analysis"]
)

# Overview & Analytics Page
if page == "📊 Overview & Analytics":
    st.header("📊 Risk Stratification Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
        st.metric("High Risk (Tier 4-5)", f"{len(df[df['risk_tier'].isin([4,5])]):,}")
    
    with col2:
        avg_age = df['age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
        st.metric("Avg Chronic Conditions", f"{df['chronic_condition_count'].mean():.1f}")
    
    with col3:
        total_costs = df['total_medicare_costs'].sum()
        st.metric("Total Medicare Costs", f"${total_costs:,.0f}")
        avg_cost = df['total_medicare_costs'].mean()
        st.metric("Avg Cost per Patient", f"${avg_cost:,.0f}")
    
    with col4:
        total_savings = df['cost_savings'].sum()
        st.metric("Potential Savings", f"${total_savings:,.0f}")
        avg_roi = df['roi_percentage'].mean()
        st.metric("Average ROI", f"{avg_roi:.1f}%")
    
    # Risk tier distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Tier Distribution")
        tier_counts = df['risk_tier'].value_counts().sort_index()
        tier_labels = df.groupby('risk_tier')['risk_tier_label'].first()
        
        fig = px.pie(
            values=tier_counts.values,
            names=[f"Tier {i}: {tier_labels[i]}" for i in tier_counts.index],
            color_discrete_sequence=['#10b981', '#06b6d4', '#f59e0b', '#ef4444', '#7c3aed']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age vs Risk Distribution")
        fig = px.scatter(
            df.sample(1000), 
            x='age', 
            y='chronic_condition_count',
            color='risk_tier',
            size='total_medicare_costs',
            color_continuous_scale='RdYlGn_r',
            labels={'chronic_condition_count': 'Chronic Conditions', 'age': 'Age'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Chronic conditions prevalence
    st.subheader("Chronic Condition Prevalence")
    condition_cols = ['SP_DIABETES', 'SP_ISCHMCHT', 'SP_RA_OA', 'SP_DEPRESSN', 'SP_OSTEOPRS',
                      'SP_CHF', 'SP_COPD', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_STRKETIA', 'SP_ALZHDMTA']
    
    condition_names = {
        'SP_DIABETES': 'Diabetes',
        'SP_ISCHMCHT': 'Ischemic Heart Disease',
        'SP_RA_OA': 'Arthritis',
        'SP_DEPRESSN': 'Depression',
        'SP_OSTEOPRS': 'Osteoporosis',
        'SP_CHF': 'Heart Failure',
        'SP_COPD': 'COPD',
        'SP_CHRNKIDN': 'Kidney Disease',
        'SP_CNCR': 'Cancer',
        'SP_STRKETIA': 'Stroke',
        'SP_ALZHDMTA': 'Alzheimer\'s'
    }
    
    condition_prevalence = df[condition_cols].mean().sort_values(ascending=True)
    condition_display_names = [condition_names[col] for col in condition_prevalence.index]
    
    fig = px.bar(
        x=condition_prevalence.values * 100,
        y=condition_display_names,
        orientation='h',
        labels={'x': 'Prevalence (%)', 'y': 'Condition'},
        title="Chronic Condition Prevalence in Patient Population"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Patient Search Page
elif page == "🔍 Patient Search":
    st.header("🔍 Patient Search & Risk Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Search Filters")
        
        # Patient ID search
        patient_id = st.text_input("Patient ID", placeholder="Enter patient ID (e.g., PAT000001)")
        
        # Risk tier filter
        risk_tiers = st.multiselect(
            "Risk Tiers",
            options=[1, 2, 3, 4, 5],
            default=[4, 5],
            format_func=lambda x: f"Tier {x}: {df[df['risk_tier']==x]['risk_tier_label'].iloc[0] if len(df[df['risk_tier']==x]) > 0 else 'Unknown'}"
        )
        
        # Age range
        age_range = st.slider("Age Range", 65, 95, (70, 85))
        
        # Chronic conditions filter
        st.subheader("Chronic Conditions")
        selected_conditions = []
        condition_names = {
            'SP_DIABETES': 'Diabetes',
            'SP_CHF': 'Heart Failure',
            'SP_COPD': 'COPD',
            'SP_CHRNKIDN': 'Kidney Disease',
            'SP_CNCR': 'Cancer'
        }
        
        for col, name in condition_names.items():
            if st.checkbox(name, key=f"filter_{col}"):
                selected_conditions.append(col)
    
    with col2:
        st.subheader("Search Results")
        
        # Apply filters
        filtered_df = df.copy()
        
        if patient_id:
            filtered_df = filtered_df[filtered_df['DESYNPUF_ID'].str.contains(patient_id, case=False, na=False)]
        
        if risk_tiers:
            filtered_df = filtered_df[filtered_df['risk_tier'].isin(risk_tiers)]
        
        filtered_df = filtered_df[
            (filtered_df['age'] >= age_range[0]) & 
            (filtered_df['age'] <= age_range[1])
        ]
        
        for condition in selected_conditions:
            filtered_df = filtered_df[filtered_df[condition] == 1]
        
        st.write(f"Found {len(filtered_df):,} patients matching criteria")
        
        if len(filtered_df) > 0:
            # Display results
            display_cols = ['DESYNPUF_ID', 'age', 'risk_tier', 'risk_tier_label',
                            'chronic_condition_count', 'total_medicare_costs', 'primary_departments']
            
            st.dataframe(
                filtered_df[display_cols].head(20),
                use_container_width=True,
                column_config={
                    'total_medicare_costs': st.column_config.NumberColumn(
                        'Total Medicare Costs',
                        format='$%.0f'
                    ),
                    'risk_tier': st.column_config.NumberColumn(
                        'Risk Tier',
                        format='%d'
                    )
                }
            )
            
            # Patient detail view
            if len(filtered_df) <= 50:  # Only show detail for smaller result sets
                selected_patient = st.selectbox(
                    "Select Patient for Detailed View",
                    options=filtered_df['DESYNPUF_ID'].tolist()
                )
                
                if selected_patient:
                    patient_data = filtered_df[filtered_df['DESYNPUF_ID'] == selected_patient].iloc[0]
                    
                    st.subheader(f"Patient Profile: {selected_patient}")
                    
                    # Patient overview
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Age", f"{patient_data['age']} years")
                        gender = "Male" if patient_data['gender_male'] == 1 else "Female"
                        st.metric("Gender", gender)
                    
                    with col2:
                        st.metric("Risk Tier", f"{patient_data['risk_tier']}")
                        st.metric("Risk Level", patient_data['risk_tier_label'])
                    
                    with col3:
                        st.metric("Chronic Conditions", f"{patient_data['chronic_condition_count']}")
                        st.metric("Hospital Admissions", f"{patient_data['inpatient_admissions']}")
                    
                    with col4:
                        st.metric("Total Medicare Costs", f"${patient_data['total_medicare_costs']:,.0f}")
                        st.metric("Potential Savings", f"${patient_data['cost_savings']:,.0f}")
                    
                    # Risk scores visualization
                    risk_scores = {
                        '30-Day Hospitalization': patient_data['risk_30d_hospitalization'],
                        '60-Day Hospitalization': patient_data['risk_60d_hospitalization'],
                        '90-Day Hospitalization': patient_data['risk_90d_hospitalization'],
                        'Mortality Risk': patient_data['mortality_risk']
                    }
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(risk_scores.keys()),
                            y=[score * 100 for score in risk_scores.values()],
                            marker_color=['#ef4444', '#f59e0b', '#06b6d4', '#7c3aed']
                        )
                    ])
                    fig.update_layout(
                        title="Risk Scores (%)",
                        yaxis_title="Risk Probability (%)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)

# New Patient Prediction Page
elif page == "➕ New Patient Prediction":
    st.header("➕ New Patient Risk Prediction")
  
    st.markdown("""
    ### 🩺 Enter Patient Information
    Complete the form below to predict hospitalization and mortality risks for a new patient.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", 65, 100, 75)
        gender_male = st.selectbox("Gender", ["Female", "Male"])
        race = st.selectbox("Race", ["White", "Black", "Other"])
        
        st.subheader("Healthcare Utilization")
        inpatient_admissions = st.number_input("Hospital Admissions (Last Year)", 0, 10, 0)
        inpatient_days = st.number_input("Total Hospital Days", 0, 100, 0)
        outpatient_visits = st.number_input("Outpatient Visits", 0, 50, 5)
    
    with col2:
        st.subheader("Chronic Conditions")
        sp_diabetes = st.checkbox("Diabetes")
        sp_chf = st.checkbox("Heart Failure")
        sp_ischmcht = st.checkbox("Ischemic Heart Disease")
        sp_copd = st.checkbox("COPD")
        sp_depressn = st.checkbox("Depression")
        sp_chrnkidn = st.checkbox("Chronic Kidney Disease")
        sp_cncr = st.checkbox("Cancer")
        sp_osteoprs = st.checkbox("Osteoporosis")
        sp_ra_oa = st.checkbox("Arthritis")
        sp_strketia = st.checkbox("Stroke History")
        sp_alzhdmta = st.checkbox("Alzheimer's/Dementia")
    
    with col3:
        st.subheader("Healthcare Costs")
        medreimb_ip = st.number_input("Inpatient Medicare Costs ($)", 0, 100000, 0)
        medreimb_op = st.number_input("Outpatient Medicare Costs ($)", 0, 50000, 2000)
        medreimb_car = st.number_input("Carrier Medicare Costs ($)", 0, 20000, 1500)
    
    # Calculate prediction button
    if st.button("🔮 Calculate Risk Prediction", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'age': age,
            'gender_male': 1 if gender_male == "Male" else 0,
            'race_white': 1 if race == "White" else 0,
            'race_black': 1 if race == "Black" else 0,
            'race_other': 1 if race == "Other" else 0,
            'age_65_74': 1 if 65 <= age < 75 else 0,
            'age_75_84': 1 if 75 <= age < 85 else 0,
            'age_85_plus': 1 if age >= 85 else 0,
            'SP_CHF': int(sp_chf),
            'SP_DIABETES': int(sp_diabetes),
            'SP_CHRNKIDN': int(sp_chrnkidn),
            'SP_CNCR': int(sp_cncr),
            'SP_COPD': int(sp_copd),
            'SP_DEPRESSN': int(sp_depressn),
            'SP_ISCHMCHT': int(sp_ischmcht),
            'SP_STRKETIA': int(sp_strketia),
            'SP_ALZHDMTA': int(sp_alzhdmta),
            'SP_OSTEOPRS': int(sp_osteoprs),
            'SP_RA_OA': int(sp_ra_oa),
            'inpatient_admissions': inpatient_admissions,
            'inpatient_days': inpatient_days,
            'outpatient_visits': outpatient_visits,
            'prior_hospitalization': 1 if inpatient_admissions > 0 else 0,
            'frequent_ed_user': 1 if outpatient_visits > 15 else 0,
            'total_medicare_costs': medreimb_ip + medreimb_op + medreimb_car,
            'high_cost_patient': 1 if (medreimb_ip + medreimb_op + medreimb_car) > 20000 else 0,
            'MEDREIMB_IP': medreimb_ip,
            'MEDREIMB_OP': medreimb_op,
            'MEDREIMB_CAR': medreimb_car
        }
        
        # Calculate chronic condition count and high-impact conditions
        chronic_conditions = [sp_diabetes, sp_chf, sp_chrnkidn, sp_cncr, sp_copd,
                             sp_depressn, sp_ischmcht, sp_strketia, sp_alzhdmta, sp_osteoprs, sp_ra_oa]
        input_data['chronic_condition_count'] = sum(chronic_conditions)
        
        high_impact = [sp_chf, sp_chrnkidn, sp_cncr, sp_copd]
        input_data['high_impact_conditions'] = sum(high_impact)
        
        # Mock prediction (since we don't have actual trained models loaded)
        # Calculate composite risk score
        age_risk = (age - 65) / 30
        condition_risk = input_data['chronic_condition_count'] / 10
        utilization_risk = min(inpatient_admissions / 5, 1)
        cost_risk = min(input_data['total_medicare_costs'] / 50000, 1)
        
        composite_risk = (age_risk * 0.2 + condition_risk * 0.4 +
                          utilization_risk * 0.25 + cost_risk * 0.15)
        
        # Generate risk predictions
        risk_30d = min(max(composite_risk * 0.6 + np.random.normal(0, 0.05), 0.01), 0.8)
        risk_60d = min(max(composite_risk * 0.7 + np.random.normal(0, 0.07), 0.02), 0.85)
        risk_90d = min(max(composite_risk * 0.8 + np.random.normal(0, 0.09), 0.03), 0.9)
        mortality_risk = min(max(composite_risk * 0.3 + np.random.normal(0, 0.04), 0.001), 0.4)
        
        # Determine risk tier
        if composite_risk >= 0.8:
            risk_tier = 5
        elif composite_risk >= 0.6:
            risk_tier = 4
        elif composite_risk >= 0.4:
            risk_tier = 3
        elif composite_risk >= 0.2:
            risk_tier = 2
        else:
            risk_tier = 1
        
        tier_info = {
            1: {'label': 'Low Risk', 'intervention': 'Preventive Care', 'cost': 150, 'color': '#10b981', 'frequency': 'Annual'},
            2: {'label': 'Low-Moderate Risk', 'intervention': 'Enhanced Wellness', 'cost': 200, 'color': '#06b6d4', 'frequency': 'Semi-Annual'},
            3: {'label': 'Moderate Risk', 'intervention': 'Care Coordination', 'cost': 300, 'color': '#f59e0b', 'frequency': 'Quarterly'},
            4: {'label': 'High Risk', 'intervention': 'Case Management', 'cost': 400, 'color': '#ef4444', 'frequency': 'Monthly'},
            5: {'label': 'Critical Risk', 'intervention': 'Intensive Management', 'cost': 500, 'color': '#7c3aed', 'frequency': 'Weekly'}
        }
        
        # Display prediction results
        st.markdown("---")
        st.subheader("🎯 Risk Prediction Results")
        
        # Risk tier display
        tier_color = tier_info[risk_tier]['color']
        st.markdown(f"""
        <div style="background: {tier_color}; color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h3>Risk Tier {risk_tier}: {tier_info[risk_tier]['label']}</h3>
            <p><strong>Recommended Intervention:</strong> {tier_info[risk_tier]['intervention']}</p>
            <p><strong>Annual Intervention Cost:</strong> ${tier_info[risk_tier]['cost']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk scores
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("30-Day Hospitalization Risk", f"{risk_30d:.1%}")
            st.metric("60-Day Hospitalization Risk", f"{risk_60d:.1%}")
            st.metric("90-Day Hospitalization Risk", f"{risk_90d:.1%}")
            st.metric("Mortality Risk", f"{mortality_risk:.1%}")
        
        with col2:
            # Risk visualization
            risk_data = {
                'Risk Type': ['30-Day Hosp.', '60-Day Hosp.', '90-Day Hosp.', 'Mortality'],
                'Probability': [risk_30d * 100, risk_60d * 100, risk_90d * 100, mortality_risk * 100]
            }
            
            fig = px.bar(
                risk_data,
                x='Risk Type',
                y='Probability',
                title="Risk Probabilities (%)",
                color='Probability',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Care recommendations
        st.subheader("📋 Recommended Care Plan")
        
        # Generate departments this patient should be assigned to
        assigned_departments = []
        if sp_chf or sp_ischmcht:
            assigned_departments.append('Cardiology')
        if sp_diabetes:
            assigned_departments.append('Endocrinology')
        if sp_copd:
            assigned_departments.append('Pulmonology')
        if sp_cncr:
            assigned_departments.append('Oncology')
        if sp_chrnkidn:
            assigned_departments.append('Nephrology')
        if sp_strketia or sp_alzhdmta:
            assigned_departments.append('Neurology')
        if sp_depressn:
            assigned_departments.append('Psychiatry')
        if sp_ra_oa or sp_osteoprs:
            assigned_departments.append('Rheumatology')
        
        if not assigned_departments or age >= 80:
            assigned_departments.append('General Medicine')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Primary Departments:**")
            for dept in assigned_departments:
                st.write(f"• {dept}")
            
            st.write(f"**Care Frequency:** {tier_info[risk_tier]['frequency']}")
            st.write(f"**Intervention Type:** {tier_info[risk_tier]['intervention']}")
        
        with col2:
            # Enhanced cost-benefit analysis for this patient
            prevention_rate = {1: 0.05, 2: 0.12, 3: 0.25, 4: 0.40, 5: 0.55}[risk_tier]
            
            # Multiple prevention streams
            prevented_hosps = risk_30d * prevention_rate
            prevented_readmissions = risk_60d * prevention_rate * 0.8
            prevented_emergency = risk_30d * prevention_rate * 2.5
            
            # Cost calculations with enhanced savings
            hosp_savings = prevented_hosps * 15000
            readmit_savings = prevented_readmissions * 10000
            emergency_savings = prevented_emergency * 2500
            quality_bonus = (hosp_savings + readmit_savings + emergency_savings) * 0.2
            
            total_savings = hosp_savings + readmit_savings + emergency_savings + quality_bonus
            net_savings = total_savings - tier_info[risk_tier]['cost']
            patient_roi = (net_savings / tier_info[risk_tier]['cost'] * 100) if tier_info[risk_tier]['cost'] > 0 else 0
            
            st.metric("Prevention Rate", f"{prevention_rate:.1%}")
            st.metric("Total Cost Savings", f"${total_savings:,.0f}")
            st.metric("Net Savings", f"${net_savings:,.0f}")
            st.metric("Patient ROI", f"{patient_roi:.1f}%")

# Department Management Page
elif page == "🏥 Department Management":
    st.header("🏥 Department-Based Risk Management")
    
   
    
    # Department analysis
    department_analysis = []
    
    # Extract all unique departments
    all_departments = []
    for dept_str in df['primary_departments']:
        all_departments.extend(dept_str.split('; '))
    
    unique_departments = list(set(all_departments))
    
    for dept in unique_departments:
        dept_patients = df[df['primary_departments'].str.contains(dept, na=False)]
        if len(dept_patients) > 50:  # Only include departments with significant patient volume
            
            high_risk_count = len(dept_patients[dept_patients['risk_tier'].isin([4, 5])])
            avg_cost = dept_patients['total_medicare_costs'].mean()
            total_savings = dept_patients['cost_savings'].sum()
            
            department_analysis.append({
                'Department': dept,
                'Total Patients': len(dept_patients),
                'High Risk Patients': high_risk_count,
                'High Risk %': (high_risk_count / len(dept_patients)) * 100,
                'Avg Medicare Cost': avg_cost,
                'Total Potential Savings': total_savings,
                'Avg Age': dept_patients['age'].mean()
            })
    
    dept_df = pd.DataFrame(department_analysis).sort_values('High Risk %', ascending=False)
    
    # Department overview table
    st.subheader("📊 Department Overview")
    st.dataframe(
        dept_df,
        use_container_width=True,
        column_config={
            'Avg Medicare Cost': st.column_config.NumberColumn('Avg Medicare Cost', format='$%.0f'),
            'Total Potential Savings': st.column_config.NumberColumn('Total Potential Savings', format='$%.0f'),
            'High Risk %': st.column_config.NumberColumn('High Risk %', format='%.1f%%'),
            'Avg Age': st.column_config.NumberColumn('Avg Age', format='%.1f')
        }
    )
    
    # Department selection
    selected_dept = st.selectbox("Select Department for Detailed Analysis", dept_df['Department'].tolist())
    
    if selected_dept:
        dept_data = dept_df[dept_df['Department'] == selected_dept].iloc[0]
        dept_patients = df[df['primary_departments'].str.contains(selected_dept, na=False)]
        
        st.subheader(f"📋 {selected_dept} Department Analysis")
        
        # Department metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", f"{dept_data['Total Patients']:,}")
            st.metric("High Risk Patients", f"{dept_data['High Risk Patients']:,}")
        
        with col2:
            st.metric("High Risk Rate", f"{dept_data['High Risk %']:.1f}%")
            st.metric("Average Age", f"{dept_data['Avg Age']:.1f} years")
        
        with col3:
            st.metric("Avg Medicare Cost", f"${dept_data['Avg Medicare Cost']:,.0f}")
            st.metric("Total Potential Savings", f"${dept_data['Total Potential Savings']:,.0f}")
        
        with col4:
            intervention_cost = dept_patients['annual_intervention_cost'].sum()
            net_savings = dept_data['Total Potential Savings'] - intervention_cost
            roi = (net_savings / intervention_cost * 100) if intervention_cost > 0 else 0
            st.metric("Intervention Cost", f"${intervention_cost:,.0f}")
            st.metric("Department ROI", f"{roi:.1f}%")
        
        # Department visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk tier distribution for department
            dept_tier_counts = dept_patients['risk_tier'].value_counts().sort_index()
            tier_labels = dept_patients.groupby('risk_tier')['risk_tier_label'].first()
            
            fig = px.pie(
                values=dept_tier_counts.values,
                names=[f"Tier {i}: {tier_labels[i]}" for i in dept_tier_counts.index],
                title=f"Risk Distribution - {selected_dept}",
                color_discrete_sequence=['#10b981', '#06b6d4', '#f59e0b', '#ef4444', '#7c3aed']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Age distribution by risk tier
            fig = px.histogram(
                dept_patients,
                x='age',
                color='risk_tier',
                title=f"Age Distribution by Risk Tier - {selected_dept}",
                color_discrete_sequence=['#10b981', '#06b6d4', '#f59e0b', '#ef4444', '#7c3aed']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Condition prevalence in department
        st.subheader(f"🩺 Common Conditions in {selected_dept}")
        condition_cols = ['SP_DIABETES', 'SP_CHF', 'SP_ISCHMCHT', 'SP_COPD', 'SP_DEPRESSN',
                          'SP_CHRNKIDN', 'SP_CNCR', 'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA', 'SP_ALZHDMTA']
        
        dept_conditions = dept_patients[condition_cols].mean().sort_values(ascending=False)
        condition_names = {
            'SP_DIABETES': 'Diabetes', 'SP_CHF': 'Heart Failure', 'SP_ISCHMCHT': 'Ischemic Heart Disease',
            'SP_COPD': 'COPD', 'SP_DEPRESSN': 'Depression', 'SP_CHRNKIDN': 'Kidney Disease',
            'SP_CNCR': 'Cancer', 'SP_OSTEOPRS': 'Osteoporosis', 'SP_RA_OA': 'Arthritis',
            'SP_STRKETIA': 'Stroke', 'SP_ALZHDMTA': 'Alzheimer\'s'
        }
        
        dept_condition_names = [condition_names[col] for col in dept_conditions.index]
        
        fig = px.bar(
            x=dept_conditions.values * 100,
            y=dept_condition_names,
            orientation='h',
            title=f"Condition Prevalence in {selected_dept} (%)",
            color=dept_conditions.values * 100,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # High-risk patients table
        st.subheader(f"🚨 High-Risk Patients in {selected_dept}")
        high_risk_dept = dept_patients[dept_patients['risk_tier'].isin([4, 5])].copy()
        
        if len(high_risk_dept) > 0:
            display_cols = ['DESYNPUF_ID', 'age', 'risk_tier', 'chronic_condition_count',
                            'total_medicare_costs', 'care_intervention', 'cost_savings']
            
            st.dataframe(
                high_risk_dept[display_cols].head(20),
                use_container_width=True,
                column_config={
                    'total_medicare_costs': st.column_config.NumberColumn('Medicare Costs', format='$%.0f'),
                    'cost_savings': st.column_config.NumberColumn('Potential Savings', format='$%.0f')
                }
            )
        else:
            st.info("No high-risk patients found in this department.")

# ROI Analysis Page
elif page == "💰 ROI Analysis":
    st.header("💰 Return on Investment Analysis")
    

    
    # Overall ROI metrics
    total_patients = len(df)
    total_intervention_cost = df['annual_intervention_cost'].sum()
    total_savings = df['cost_savings'].sum()
    quality_bonus = total_savings * 0.15  # Quality improvements
    total_benefits = total_savings + quality_bonus
    net_savings = total_benefits - total_intervention_cost
    overall_roi = (net_savings / total_intervention_cost * 100) if total_intervention_cost > 0 else 0
    
    # ROI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Patients", f"{total_patients:,}")
    with col2:
        st.metric("Intervention Cost", f"${total_intervention_cost:,.0f}")
    with col3:
        st.metric("Direct Savings", f"${total_savings:,.0f}")
    with col4:
        st.metric("Total Benefits", f"${total_benefits:,.0f}")
    with col5:
        st.metric("Net Savings & ROI", f"${net_savings:,.0f}", f"{overall_roi:.1f}%")
    
    # ROI by Risk Tier
    st.subheader("📊 ROI Analysis by Risk Tier")
    
    tier_analysis = []
    for tier in range(1, 6):
        tier_data = df[df['risk_tier'] == tier]
        if len(tier_data) > 0:
            tier_intervention_cost = tier_data['annual_intervention_cost'].sum()
            tier_direct_savings = tier_data['cost_savings'].sum()
            tier_quality_bonus = tier_direct_savings * 0.15
            tier_total_benefits = tier_direct_savings + tier_quality_bonus
            tier_net = tier_total_benefits - tier_intervention_cost
            tier_roi = (tier_net / tier_intervention_cost * 100) if tier_intervention_cost > 0 else 0
            
            tier_analysis.append({
                'Risk Tier': tier,
                'Patients': len(tier_data),
                'Intervention Cost': tier_intervention_cost,
                'Direct Savings': tier_direct_savings,
                'Quality Bonus': tier_quality_bonus,
                'Total Benefits': tier_total_benefits,
                'Net Savings': tier_net,
                'ROI %': tier_roi
            })
    
    roi_df = pd.DataFrame(tier_analysis)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI by tier bar chart
        fig = px.bar(
            roi_df,
            x='Risk Tier',
            y='ROI %',
            title="ROI Percentage by Risk Tier",
            color='ROI %',
            color_continuous_scale='RdYlGn',
            text='ROI %'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cost vs Benefits comparison
        fig = go.Figure(data=[
            go.Bar(name='Intervention Cost', x=roi_df['Risk Tier'], y=roi_df['Intervention Cost'],
                   marker_color='lightcoral', text=roi_df['Intervention Cost'], texttemplate='$%{text:,.0f}'),
            go.Bar(name='Total Benefits', x=roi_df['Risk Tier'], y=roi_df['Total Benefits'],
                   marker_color='lightgreen', text=roi_df['Total Benefits'], texttemplate='$%{text:,.0f}')
        ])
        fig.update_layout(
            barmode='group',
            title="Intervention Costs vs Total Benefits by Tier",
            xaxis_title="Risk Tier",
            yaxis_title="Amount ($)",
            height=400
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed ROI table
    st.subheader("📋 Detailed ROI Breakdown")
    st.dataframe(
        roi_df,
        use_container_width=True,
        column_config={
            'Intervention Cost': st.column_config.NumberColumn('Intervention Cost', format='$%.0f'),
            'Direct Savings': st.column_config.NumberColumn('Direct Savings', format='$%.0f'),
            'Quality Bonus': st.column_config.NumberColumn('Quality Bonus', format='$%.0f'),
            'Total Benefits': st.column_config.NumberColumn('Total Benefits', format='$%.0f'),
            'Net Savings': st.column_config.NumberColumn('Net Savings', format='$%.0f'),
            'ROI %': st.column_config.NumberColumn('ROI %', format='%.1f%%')
        }
    )
    
    # Time-based ROI projections
    st.subheader("📈 5-Year ROI Projection")
    
    years = list(range(1, 6))
    cumulative_savings = []
    cumulative_costs = []
    
    # Assume 3% annual cost inflation and 5% improvement in intervention effectiveness
    for year in years:
        inflation_factor = 1.03 ** year
        effectiveness_factor = 1.05 ** year
        
        annual_cost = total_intervention_cost * inflation_factor
        annual_benefits = total_benefits * effectiveness_factor
        
        if year == 1:
            cumulative_savings.append(annual_benefits)
            cumulative_costs.append(annual_cost)
        else:
            cumulative_savings.append(cumulative_savings[-1] + annual_benefits)
            cumulative_costs.append(cumulative_costs[-1] + annual_cost)
    
    projection_df = pd.DataFrame({
        'Year': years,
        'Cumulative Intervention Cost': cumulative_costs,
        'Cumulative Benefits': cumulative_savings,
        'Net Savings': [s - c for s, c in zip(cumulative_savings, cumulative_costs)],
        'ROI %': [(s - c) / c * 100 for s, c in zip(cumulative_savings, cumulative_costs)]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            projection_df,
            x='Year',
            y=['Cumulative Intervention Cost', 'Cumulative Benefits'],
            title="5-Year Cumulative Cost vs Benefits",
            labels={'value': 'Amount ($)', 'variable': 'Metric'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            projection_df,
            x='Year',
            y='ROI %',
            title="5-Year ROI Progression",
            labels={'ROI %': 'ROI Percentage (%)', 'Year': 'Year'},
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Financial impact summary
    st.subheader("💸 Financial Impact Summary")
    
    final_roi = projection_df['ROI %'].iloc[-1]
    total_5yr_net = projection_df['Net Savings'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("5-Year Net Savings", f"${total_5yr_net:,.0f}")
    with col2:
        st.metric("5-Year ROI", f"{final_roi:.1f}%")
    with col3:
        payback_period = "Year 1" if projection_df['Net Savings'].iloc[0] > 0 else "Year 2"
        st.metric("Payback Period", payback_period)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🏥 Medicare Risk Prediction Dashboard | Built with Streamlit & Advanced ML Models</p>
    <p>Synthetic data for demonstration purposes | Based on CMS SynPUF methodology</p>
</div>
""", unsafe_allow_html=True)

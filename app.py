"""
Customer Segmentation & Churn Pattern Analytics in European Banking
Interactive Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import load_and_preprocess_data
from src.analytics_engine import (
    compute_overall_kpis,
    compute_segment_summary,
    compute_geographic_risk_index,
    ChurnPredictor
)

# -------------------------------------------------------------
# 1. Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="European Banking Churn & Customer Segmentation Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Styling */
    .main-header {
        background: linear-gradient(135deg, #0e2a47 0%, #1e3a8a 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 15px;
        color: #cbd5e1;
    }
    /* Metric Cards */
    .kpi-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 18px 20px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    .kpi-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 2px;
    }
    /* Section Headers */
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 15px;
        margin-bottom: 15px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
    }
    /* Insight Alert Box */
    .insight-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 18px;
        font-size: 14px;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Data Loading & Caching
# -------------------------------------------------------------
@st.cache_data(show_spinner="Loading European Banking Portfolio Data...")
def get_data():
    return load_and_preprocess_data()

@st.cache_resource(show_spinner="Initializing Churn Prediction Engine...")
def get_model(df):
    predictor = ChurnPredictor()
    predictor.train(df)
    return predictor

try:
    df_raw = get_data()
    predictor = get_model(df_raw)
except Exception as e:
    st.error(f"Error initializing data or model: {e}")
    st.stop()

# -------------------------------------------------------------
# 3. Sidebar Filters
# -------------------------------------------------------------
st.sidebar.markdown("### 🏦 **Filter Portfolio**")
st.sidebar.caption("Slice customer segments across demographic and financial dimensions.")

# Reset Button
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    for key in list(st.session_state.keys()):
        if key.startswith("filter_"):
            del st.session_state[key]
    st.rerun()

# 1. Geography
all_geos = sorted(df_raw['Geography'].unique())
selected_geos = st.sidebar.multiselect(
    "European Region / Country",
    options=all_geos,
    default=all_geos,
    key="filter_geo"
)

# 2. Gender
all_genders = ['All', 'Female', 'Male']
selected_gender = st.sidebar.selectbox("Gender", options=all_genders, index=0, key="filter_gender")

# 3. Age Groups
all_age_groups = list(df_raw['Age_Group'].cat.categories)
selected_age_groups = st.sidebar.multiselect(
    "Age Segment",
    options=all_age_groups,
    default=all_age_groups,
    key="filter_age_group"
)

# 4. Credit Score Bands
all_cs_bands = list(df_raw['CreditScore_Band'].cat.categories)
selected_cs_bands = st.sidebar.multiselect(
    "Credit Score Band",
    options=all_cs_bands,
    default=all_cs_bands,
    key="filter_cs_band"
)

# 5. Balance Segment
all_bal_segs = list(df_raw['Balance_Segment'].cat.categories)
selected_bal_segs = st.sidebar.multiselect(
    "Account Balance Segment",
    options=all_bal_segs,
    default=all_bal_segs,
    key="filter_bal_seg"
)

# 6. Number of Products
all_prods = list(df_raw['Products_Segment'].cat.categories)
selected_prods = st.sidebar.multiselect(
    "Products Held",
    options=all_prods,
    default=all_prods,
    key="filter_prods"
)

# 7. Member Activity
activity_filter = st.sidebar.radio(
    "Engagement Status",
    options=["All Customers", "Active Members Only", "Inactive Members Only"],
    index=0,
    key="filter_activity"
)

# Apply Filters
df_filtered = df_raw.copy()

if selected_geos:
    df_filtered = df_filtered[df_filtered['Geography'].isin(selected_geos)]
if selected_gender != 'All':
    df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]
if selected_age_groups:
    df_filtered = df_filtered[df_filtered['Age_Group'].isin(selected_age_groups)]
if selected_cs_bands:
    df_filtered = df_filtered[df_filtered['CreditScore_Band'].isin(selected_cs_bands)]
if selected_bal_segs:
    df_filtered = df_filtered[df_filtered['Balance_Segment'].isin(selected_bal_segs)]
if selected_prods:
    df_filtered = df_filtered[df_filtered['Products_Segment'].isin(selected_prods)]
if activity_filter == "Active Members Only":
    df_filtered = df_filtered[df_filtered['IsActiveMember'] == 1]
elif activity_filter == "Inactive Members Only":
    df_filtered = df_filtered[df_filtered['IsActiveMember'] == 0]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Portfolio**: `{len(df_filtered):,}` / `{len(df_raw):,}` accounts ({round(len(df_filtered)/len(df_raw)*100, 1)}%)")

# -------------------------------------------------------------
# 4. Header Banner
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Customer Segmentation & Churn Pattern Analytics in European Banking</h1>
    <p>Empirical Portfolio Analytics, Supervisory Risk Diagnostics, and Segment-Driven Retention Architecture across France, Germany, and Spain</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 5. Core Navigation Tabs
# -------------------------------------------------------------
tab_exec, tab_geo, tab_demo, tab_high_val, tab_sim = st.tabs([
    "📊 Executive Summary & KPIs",
    "🌍 Geographic Churn Patterns",
    "👥 Demographics & Tenure",
    "💎 High-Value Customer & Capital Risk",
    "🔮 Risk Scoring & Retention Simulator"
])

# Compute dynamic filtered KPIs
kpis = compute_overall_kpis(df_filtered)
base_kpis = compute_overall_kpis(df_raw)

# -------------------------------------------------------------
# TAB 1: Executive Summary & KPIs
# -------------------------------------------------------------
with tab_exec:
    st.markdown("<div class='section-title'>Key Performance Indicators (KPIs)</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        churn_delta = round(kpis['churn_rate'] - base_kpis['churn_rate'], 2)
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #ef4444;">
            <div class="kpi-title">Overall Churn Rate</div>
            <div class="kpi-value">{kpis['churn_rate']}%</div>
            <div class="kpi-subtext">Baseline: {base_kpis['churn_rate']}% ({'+' if churn_delta >= 0 else ''}{churn_delta}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #3b82f6;">
            <div class="kpi-title">Total Active Base</div>
            <div class="kpi-value">{kpis['total_customers']:,}</div>
            <div class="kpi-subtext">Churned: {kpis['churned_count']:,} | Retained: {kpis['retained_count']:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #f59e0b;">
            <div class="kpi-title">High-Value Churn Rate</div>
            <div class="kpi-value">{kpis['high_val_churn_rate']}%</div>
            <div class="kpi-subtext">Premium accounts (>€100k)</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #dc2626;">
            <div class="kpi-title">Capital at Risk (Lost)</div>
            <div class="kpi-value">€{kpis['total_capital_at_risk']/1e6:.1f}M</div>
            <div class="kpi-subtext">Total churned deposits volume</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #8b5cf6;">
            <div class="kpi-title">Inactivity Churn Multiplier</div>
            <div class="kpi-value">{kpis['engagement_drop_ratio']}x</div>
            <div class="kpi-subtext">Inactive ({kpis['inactive_churn_rate']}%) vs Active ({kpis['active_churn_rate']}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <b>Executive Supervisory Insight</b>: Portfolio churn is heavily concentrated along three critical structural fault lines:
        <b>(1) German retail branches</b> exhibit a <b>39.9% churn rate</b>, more than double France (16.2%) and Spain (16.7%);
        <b>(2) Demographic vulnerability</b> peaks in the <b>46–60 pre-retirement bracket (56.2% churn)</b>; and
        <b>(3) The Multi-Product Paradox</b>: while 2-product holders have optimal retention (7.6% churn), holders of 3 or 4 products experience catastrophic attrition (>82%), signaling severe customer friction or rate-shopping.
    </div>
    """, unsafe_allow_html=True)

    # Overview Visualizations
    col_v1, col_v2, col_v3 = st.columns([1, 1.2, 1.2])

    with col_v1:
        st.markdown("##### Portfolio Retention Status")
        donut_fig = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[kpis['retained_count'], kpis['churned_count']],
            hole=.55,
            marker_colors=['#10b981', '#ef4444'],
            textinfo='percent+label'
        )])
        donut_fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300, showlegend=False)
        st.plotly_chart(donut_fig, use_container_width=True)

    with col_v2:
        st.markdown("##### Regional Churn Rate vs Portfolio Benchmark")
        geo_summary = compute_geographic_risk_index(df_filtered)
        if not geo_summary.empty:
            bar_geo = px.bar(
                geo_summary,
                x='Geography',
                y='Churn_Rate_Pct',
                color='Geography',
                color_discrete_map={'France': '#3b82f6', 'Germany': '#ef4444', 'Spain': '#f59e0b'},
                text='Churn_Rate_Pct'
            )
            bar_geo.add_hline(y=base_kpis['churn_rate'], line_dash="dash", line_color="black", annotation_text=f"Benchmark ({base_kpis['churn_rate']}%)")
            bar_geo.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            bar_geo.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Churn Rate (%)", showlegend=False)
            st.plotly_chart(bar_geo, use_container_width=True)

    with col_v3:
        st.markdown("##### Churn Rate by Product Holdings")
        prod_summary = compute_segment_summary(df_filtered, 'Products_Segment')
        if not prod_summary.empty:
            bar_prod = px.bar(
                prod_summary,
                x='Products_Segment',
                y='Churn_Rate_Pct',
                color='Products_Segment',
                color_discrete_sequence=['#60a5fa', '#10b981', '#ef4444'],
                text='Churn_Rate_Pct'
            )
            bar_prod.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            bar_prod.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="Churn Rate (%)", showlegend=False)
            st.plotly_chart(bar_prod, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: Geographic Churn Patterns
# -------------------------------------------------------------
with tab_geo:
    st.markdown("<div class='section-title'>Geographic Churn Exposure & The German Paradox</div>", unsafe_allow_html=True)

    geo_table = compute_geographic_risk_index(df_filtered)
    
    g_col1, g_col2, g_col3 = st.columns(3)
    if not geo_table.empty:
        for idx, row in geo_table.iterrows():
            col = [g_col1, g_col2, g_col3][idx % 3]
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="border-left-color: {'#ef4444' if row['Churn_Rate_Pct'] > 25 else '#3b82f6'};">
                    <div class="kpi-title">{row['Geography']} Market Exposure</div>
                    <div class="kpi-value">{row['Churn_Rate_Pct']}%</div>
                    <div class="kpi-subtext">
                        <b>Risk Index</b>: {row.get('Geographic_Risk_Index', 'N/A')} | 
                        <b>Accounts</b>: {row['Total_Customers']:,} ({row['Segment_Share_Pct']}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    col_g1, col_g2 = st.columns([1.3, 1])
    with col_g1:
        st.markdown("##### Cross-Country Structural Metrics Comparison")
        comp_df = df_filtered.groupby('Geography', observed=True).agg(
            Total_Customers=('CustomerId', 'count'),
            Churn_Rate=('Exited', lambda x: np.round(x.mean() * 100, 1)),
            Avg_Balance=('Balance', lambda x: np.round(x.mean(), 0)),
            Avg_Salary=('EstimatedSalary', lambda x: np.round(x.mean(), 0)),
            Avg_Age=('Age', lambda x: np.round(x.mean(), 1)),
            Active_Pct=('IsActiveMember', lambda x: np.round(x.mean() * 100, 1)),
            Zero_Balance_Pct=('Balance', lambda x: np.round((x == 0).mean() * 100, 1))
        ).reset_index()

        st.dataframe(
            comp_df.style.format({
                'Total_Customers': '{:,}',
                'Churn_Rate': '{:.1f}%',
                'Avg_Balance': '€{:,.0f}',
                'Avg_Salary': '€{:,.0f}',
                'Avg_Age': '{:.1f} yrs',
                'Active_Pct': '{:.1f}%',
                'Zero_Balance_Pct': '{:.1f}%'
            }).background_gradient(subset=['Churn_Rate'], cmap='Reds'),
            use_container_width=True
        )

        st.markdown("""
        <div class="insight-box">
            <b>Key Finding: The German Deposit Concentration</b>:
            In France and Spain, over <b>50% of retail customers hold a €0 balance</b>, skewing average churn. 
            In contrast, <b>100% of German customers carry substantial active balances</b> (averaging <b>€119,730</b>).
            Because German customers maintain higher liquid balances, their <b>39.9% churn rate constitutes a systemic capital flight risk</b> representing over €168M in churned deposits.
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        st.markdown("##### Total Capital Lost by Country (€ Millions)")
        cap_by_geo = df_filtered[df_filtered['Exited'] == 1].groupby('Geography', observed=True)['Balance'].sum().reset_index()
        cap_by_geo['Balance_M'] = cap_by_geo['Balance'] / 1e6
        fig_cap = px.pie(
            cap_by_geo,
            names='Geography',
            values='Balance_M',
            color='Geography',
            color_discrete_map={'France': '#3b82f6', 'Germany': '#ef4444', 'Spain': '#f59e0b'},
            hole=.4
        )
        fig_cap.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_cap, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Demographics & Tenure Patterns
# -------------------------------------------------------------
with tab_demo:
    st.markdown("<div class='section-title'>Demographic Drivers & Tenure Analysis</div>", unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("##### Age Group vs Churn Rate (%): The Pre-Retirement Peak")
        age_summary = compute_segment_summary(df_filtered, 'Age_Group')
        if not age_summary.empty:
            fig_age = px.bar(
                age_summary,
                x='Age_Group',
                y='Churn_Rate_Pct',
                text='Churn_Rate_Pct',
                color='Churn_Rate_Pct',
                color_continuous_scale='Reds'
            )
            fig_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_age.update_layout(height=340, yaxis_title="Churn Rate (%)", coloraxis_showscale=False)
            st.plotly_chart(fig_age, use_container_width=True)

    with col_d2:
        st.markdown("##### Age Distribution: Retained vs Churned Customers")
        fig_hist = px.histogram(
            df_filtered,
            x='Age',
            color='Churn_Label',
            barmode='overlay',
            nbins=35,
            color_discrete_map={'Retained': '#10b981', 'Churned': '#ef4444'}
        )
        fig_hist.update_layout(height=340, yaxis_title="Customer Count", xaxis_title="Age (Years)")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    col_d3, col_d4 = st.columns(2)

    with col_d3:
        st.markdown("##### Gender Churn Disparities by Country")
        gender_geo = df_filtered.groupby(['Geography', 'Gender'], observed=True)['Exited'].agg(['count', 'mean']).reset_index()
        gender_geo['Churn_Rate_Pct'] = np.round(gender_geo['mean'] * 100, 1)

        fig_gg = px.bar(
            gender_geo,
            x='Geography',
            y='Churn_Rate_Pct',
            color='Gender',
            barmode='group',
            color_discrete_map={'Female': '#ec4899', 'Male': '#3b82f6'},
            text='Churn_Rate_Pct'
        )
        fig_gg.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_gg.update_layout(height=320, yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig_gg, use_container_width=True)

    with col_d4:
        st.markdown("##### Tenure Cohort Stability (Years with Bank)")
        tenure_summary = compute_segment_summary(df_filtered, 'Tenure')
        if not tenure_summary.empty:
            fig_tenure = px.line(
                tenure_summary,
                x='Tenure',
                y='Churn_Rate_Pct',
                markers=True,
                line_shape='spline'
            )
            fig_tenure.update_traces(line_color='#2563eb', line_width=3, marker=dict(size=8, color='#ef4444'))
            fig_tenure.add_hline(y=base_kpis['churn_rate'], line_dash="dash", line_color="gray", annotation_text="Baseline")
            fig_tenure.update_layout(height=320, yaxis_title="Churn Rate (%)", xaxis_title="Tenure (Years)")
            st.plotly_chart(fig_tenure, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: High-Value Customer & Capital Risk
# -------------------------------------------------------------
with tab_high_val:
    st.markdown("<div class='section-title'>High-Value Customer Exposure & Capital at Risk</div>", unsafe_allow_html=True)

    hv_total = len(df_filtered[df_filtered['Is_High_Value'] == True])
    hv_churned = len(df_filtered[(df_filtered['Is_High_Value'] == True) & (df_filtered['Exited'] == 1)])
    hv_churn_pct = round((hv_churned / hv_total * 100), 1) if hv_total > 0 else 0.0

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #f59e0b;">
            <div class="kpi-title">High-Value Customer Base</div>
            <div class="kpi-value">{hv_total:,}</div>
            <div class="kpi-subtext">Accounts with balance ≥ €100k or high liquid net worth</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #ef4444;">
            <div class="kpi-title">High-Value Churn Rate</div>
            <div class="kpi-value">{hv_churn_pct}%</div>
            <div class="kpi-subtext">Compared to standard customer churn of {kpis['churn_rate']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #dc2626;">
            <div class="kpi-title">High-Value Lost Capital</div>
            <div class="kpi-value">€{kpis['high_val_capital_at_risk']/1e6:.1f}M</div>
            <div class="kpi-subtext">Direct deposit balance attrition</div>
        </div>
        """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns([1.2, 1])

    with col_p1:
        st.markdown("##### Account Balance vs Estimated Salary (by Churn Status)")
        sample_size = min(2000, len(df_filtered))
        scatter_fig = px.scatter(
            df_filtered.sample(sample_size, random_state=42),
            x='Balance',
            y='EstimatedSalary',
            color='Churn_Label',
            opacity=0.6,
            color_discrete_map={'Retained': '#10b981', 'Churned': '#ef4444'},
            hover_data=['CustomerId', 'Geography', 'Age', 'NumOfProducts']
        )
        scatter_fig.add_vline(x=100000, line_dash="dash", line_color="#d97706", annotation_text="High-Value Cutoff (€100k)")
        scatter_fig.update_layout(height=380, xaxis_title="Account Balance (€)", yaxis_title="Estimated Salary (€)")
        st.plotly_chart(scatter_fig, use_container_width=True)

    with col_p2:
        st.markdown("##### The Multi-Product Retention Paradox")
        prod_data = df_filtered.groupby('NumOfProducts', observed=True).agg(
            Total=('CustomerId', 'count'),
            Churn_Rate=('Exited', lambda x: np.round(x.mean() * 100, 1))
        ).reset_index()

        fig_p = px.bar(
            prod_data,
            x='NumOfProducts',
            y='Churn_Rate',
            text='Churn_Rate',
            color='Churn_Rate',
            color_continuous_scale='Turbo'
        )
        fig_p.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_p.update_layout(height=380, yaxis_title="Churn Rate (%)", xaxis_title="Number of Products Held", coloraxis_showscale=False)
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("##### 📋 High-Risk, High-Value Accounts Drilldown Table")
    st.caption("Top vulnerable accounts prioritized by account balance and churn risk profile.")
    
    high_risk_hv = df_filtered[
        (df_filtered['Is_High_Value'] == True) & 
        (df_filtered['Exited'] == 1)
    ][['CustomerId', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'EstimatedSalary']]
    
    st.dataframe(
        high_risk_hv.head(20).style.format({
            'Balance': '€{:,.2f}',
            'EstimatedSalary': '€{:,.2f}'
        }),
        use_container_width=True
    )

    csv_data = high_risk_hv.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export At-Risk High-Value Accounts to CSV",
        data=csv_data,
        file_name="at_risk_high_value_accounts.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------
# TAB 5: Predictive Risk Scoring & Retention Simulator
# -------------------------------------------------------------
with tab_sim:
    st.markdown("<div class='section-title'>Interactive Churn Risk Scoring & Retention Simulator</div>", unsafe_allow_html=True)
    st.caption("Evaluate customer risk profile using the machine learning engine and generate targeted retention strategies.")

    col_m1, col_m2 = st.columns([1, 1.4])

    with col_m1:
        st.markdown("##### Model Performance & Feature Drivers")
        st.info(f"Classifier: **Balanced Random Forest** | Model ROC-AUC: **{predictor.auc_score:.3f}** | Accuracy: **{predictor.acc_score*100:.1f}%**")

        if predictor.feature_importances_ is not None:
            fig_fi = px.bar(
                predictor.feature_importances_.head(8),
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig_fi.update_layout(height=320, yaxis=dict(autorange="reversed"), coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_fi, use_container_width=True)

    with col_m2:
        st.markdown("##### 👤 Customer Profile Simulation Form")
        with st.form("risk_eval_form"):
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                sim_geo = st.selectbox("Country / Geography", options=["France", "Germany", "Spain"], index=1)
                sim_gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
                sim_age = st.slider("Customer Age", min_value=18, max_value=85, value=48)
                sim_tenure = st.slider("Tenure with Bank (Years)", min_value=0, max_value=10, value=3)
                sim_credit = st.number_input("Credit Score", min_value=350, max_value=850, value=650)
            with c_s2:
                sim_balance = st.number_input("Account Balance (€)", min_value=0.0, max_value=300000.0, value=125000.0, step=5000.0)
                sim_salary = st.number_input("Estimated Annual Salary (€)", min_value=10000.0, max_value=250000.0, value=85000.0, step=5000.0)
                sim_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=0)
                sim_active = st.checkbox("Active Member in Last 12 Months", value=False)
                sim_card = st.checkbox("Holds Bank Credit Card", value=True)

            btn_sim = st.form_submit_button("⚡ Predict Churn Probability & Strategy", use_container_width=True)

        if btn_sim:
            pred_res = predictor.predict_single(
                credit_score=sim_credit,
                age=sim_age,
                tenure=sim_tenure,
                balance=sim_balance,
                num_products=sim_products,
                has_card=sim_card,
                is_active=sim_active,
                salary=sim_salary,
                geo=sim_geo,
                gender=sim_gender
            )

            st.markdown(f"""
            <div style="background-color: {pred_res['color']}15; border: 2px solid {pred_res['color']}; border-radius: 10px; padding: 20px; margin-top: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 14px; font-weight: 600; text-transform: uppercase; color: {pred_res['color']};">Risk Classification</span>
                        <h2 style="margin: 2px 0 0 0; color: {pred_res['color']};">{pred_res['risk_tier']} ({pred_res['probability']}%)</h2>
                    </div>
                    <div style="font-size: 32px;">{'🚨' if pred_res['probability'] >= 50 else '✅'}</div>
                </div>
                <hr style="margin: 12px 0; border-color: {pred_res['color']}40;" />
                <div style="font-size: 14px; color: #1e293b;">
                    <b>Recommended Branch Retention Action</b>:<br/>
                    {pred_res['recommended_action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("European Banking Customer Segmentation & Churn Analytics System | Data & Models based on European Central Bank Retail Portfolio Analytics Standards.")

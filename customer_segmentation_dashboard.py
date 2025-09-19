import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import altair as alt

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the fictional retail dataset"""
    return pd.read_csv(path)

def compute_clusters(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Compute KMeans clusters on selected numerical features and return
    the DataFrame with an added 'Cluster' column (string).
    """
    # Define numerical features for clustering
    features = [
        'Age', 'Income', 'PurchaseFrequency', 'AnnualSpending',
        'Tenure', 'LastPurchaseDays', 'LoyaltyScore', 'CLV'
    ]
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X)
    # Add cluster as string for categorical encoding
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters.astype(str)
    return df_clustered

def cluster_summary(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each cluster"""
    summary = (
        df_clustered.groupby('Cluster')
        .agg(
            Size=('Cluster', 'count'),
            AvgAge=('Age', 'mean'),
            AvgIncome=('Income', 'mean'),
            AvgAnnualSpending=('AnnualSpending', 'mean'),
            AvgPurchaseFreq=('PurchaseFrequency', 'mean'),
            AvgTenure=('Tenure', 'mean'),
            AvgLoyalty=('LoyaltyScore', 'mean'),
            AvgCLV=('CLV', 'mean')
        )
        .reset_index()
    )
    return summary

def main():
    st.set_page_config(page_title="Customer Segmentation Dashboard - create by Ahmed Gbadamassi", layout="wide")
    st.title("Customer Segmentation Dashboard - create by Ahmed Gbadamassi")
    st.markdown(
        """
        This dashboard allows you to explore customer segments generated from a fictional retail dataset.
        Adjust the number of clusters to see how customers group together based on demographics, purchase behavior and loyalty metrics.
        """
    )
    # Load data
    data_path = "fictional_retail_data.csv"
    df = load_data(data_path)
    # Sidebar controls
    st.sidebar.header("Segmentation Controls")
    num_clusters = st.sidebar.slider(
        "Number of clusters (K)", min_value=2, max_value=8, value=4, step=1
    )
    # Compute clusters
    df_clustered = compute_clusters(df, num_clusters)
    # Summary table
    summary = cluster_summary(df_clustered)
    st.subheader("Cluster Summary Statistics")
    st.dataframe(summary.style.format({
        'AvgAge': '{:.1f}', 'AvgIncome': '{:.2f}', 'AvgAnnualSpending': '{:.2f}',
        'AvgPurchaseFreq': '{:.2f}', 'AvgTenure': '{:.2f}', 'AvgLoyalty': '{:.2f}', 'AvgCLV': '{:.2f}'
    }))
    # Scatter plot Age vs Annual Spending
    st.subheader("Age vs. Annual Spending")
    scatter = (
        alt.Chart(df_clustered)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X('Age', title='Age'),
            y=alt.Y('AnnualSpending', title='Annual Spending (USD)'),
            color=alt.Color('Cluster:N', title='Cluster'),
            tooltip=['Cluster', 'Age', 'AnnualSpending', 'Income', 'PurchaseFrequency']
        )
        .properties(width=550, height=350)
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)
    # Bar chart: Average CLV per cluster
    st.subheader("Average Customer Lifetime Value (CLV) per Cluster")
    bar_data = summary.copy()
    bar_chart = (
        alt.Chart(bar_data)
        .mark_bar()
        .encode(
            x=alt.X('Cluster:N', title='Cluster'),
            y=alt.Y('AvgCLV:Q', title='Average CLV (USD)'),
            color=alt.Color('Cluster:N', legend=None)
        )
        .properties(width=550, height=350)
    )
    st.altair_chart(bar_chart, use_container_width=True)
    # Category distribution stacked bar per cluster
    st.subheader("Preferred Product Category Distribution by Cluster")
    # Compute proportions
    crosstab = pd.crosstab(df_clustered['Cluster'], df_clustered['PreferredCategory'])
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0)
    crosstab_pct = crosstab_pct.reset_index().melt(id_vars='Cluster', var_name='Category', value_name='Proportion')
    stacked_bar = (
        alt.Chart(crosstab_pct)
        .mark_bar()
        .encode(
            x=alt.X('Cluster:N', title='Cluster'),
            y=alt.Y('Proportion:Q', stack='normalize', title='Proportion'),
            color=alt.Color('Category:N', title='Product Category')
        )
        .properties(width=550, height=350)
    )
    st.altair_chart(stacked_bar, use_container_width=True)
    # Data preview and download option
    with st.expander("Show raw data"):
        st.dataframe(df_clustered)
        csv = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download clustered data as CSV",
            data=csv,
            file_name='clustered_retail_data.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()

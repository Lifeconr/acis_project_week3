import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import os

# Ensure plots directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

def load_data(file_path):
    """Load and preprocess insurance dataset."""
    df = pd.read_csv(file_path)
    # Convert TransactionMonth to datetime
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], format='%Y-%m')
    # Convert categorical columns to category type
    categorical_cols = ['Province', 'Gender', 'VehicleType', 'CoverType', 'Make', 'Model']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

def summarize_data(df):
    """Calculate descriptive statistics and data types."""
    numerical_stats = df.select_dtypes(include=['float64', 'int64']).describe()
    data_types = df.dtypes
    return numerical_stats, data_types

def check_missing_values(df):
    """Identify missing values in the dataset."""
    missing = df.isnull().sum()
    return missing[missing > 0]

def calculate_loss_ratio(df):
    """Calculate Loss Ratio (TotalClaims / TotalPremium)."""
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)  # Avoid division by zero
    return df

def plot_univariate_numerical(df, column, title, filename):
    """Plot histogram for numerical column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_univariate_categorical(df, column, title, filename):
    """Plot bar chart for categorical column."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_loss_ratio_by_group(df, group_col, title, filename):
    """Plot average Loss Ratio by group (e.g., Province, Gender)."""
    loss_by_group = df.groupby(group_col)['LossRatio'].mean().sort_values()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=loss_by_group.index, y=loss_by_group.values)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel('Average Loss Ratio')
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_temporal_trends(df):
    """Plot claim frequency and severity over time."""
    claims_over_time = df.groupby('TransactionMonth').agg({
        'TotalClaims': ['count', 'mean']
    }).reset_index()
    claims_over_time.columns = ['TransactionMonth', 'ClaimCount', 'AvgClaimAmount']
    
    plt.figure(figsize=(12, 6))
    plt.plot(claims_over_time['TransactionMonth'], claims_over_time['ClaimCount'], label='Claim Frequency')
    plt.plot(claims_over_time['TransactionMonth'], claims_over_time['AvgClaimAmount'], label='Avg Claim Amount')
    plt.title('Claim Trends Over Time')
    plt.xlabel('Transaction Month')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('plots/claim_trends.png')
    plt.close()

def plot_outliers(df, column, title, filename):
    """Plot box plot to detect outliers."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(title)
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_creative_insights(df):
    """Generate three creative visualizations."""
    # Plot 1: Loss Ratio by Province and Gender
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Province', y='LossRatio', hue='Gender', data=df)
    plt.title('Loss Ratio by Province and Gender')
    plt.xticks(rotation=45)
    plt.savefig('plots/loss_ratio_province_gender.png')
    plt.close()
    
    # Plot 2: Claim Severity by Vehicle Type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='VehicleType', y='TotalClaims', data=df)
    plt.title('Claim Severity by Vehicle Type')
    plt.xticks(rotation=45)
    plt.savefig('plots/claim_severity_vehicle_type.png')
    plt.close()
    
    # Plot 3: Heatmap of Claims by Province and Month
    claims_by_month_province = df.pivot_table(values='TotalClaims', index='Province', 
                                             columns='TransactionMonth', aggfunc='sum')
    plt.figure(figsize=(14, 8))
    sns.heatmap(claims_by_month_province, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Total Claims by Province and Month')
    plt.savefig('plots/claims_heatmap.png')
    plt.close()

def perform_statistical_tests(df, group_col='Province'):
    """Perform ANOVA to test Loss Ratio differences across groups."""
    loss_by_group = [df[df[group_col] == g]['LossRatio'].dropna() for g in df[group_col].unique()]
    f_stat, p_value = f_oneway(*loss_by_group)
    return f_stat, p_value

def main(file_path):
    """Main function to orchestrate EDA."""
    # Load and preprocess data
    df = load_data(file_path)
    
    # Summarize data
    numerical_stats, data_types = summarize_data(df)
    print("Numerical Statistics:\n", numerical_stats)
    print("\nData Types:\n", data_types)
    
    # Check missing values
    missing = check_missing_values(df)
    print("\nMissing Values:\n", missing)
    
    # Calculate Loss Ratio
    df = calculate_loss_ratio(df)
    
    # Univariate analysis
    plot_univariate_numerical(df, 'TotalPremium', 'Distribution of Total Premium', 'total_premium.png')
    plot_univariate_numerical(df, 'TotalClaims', 'Distribution of Total Claims', 'total_claims.png')
    plot_univariate_categorical(df, 'Province', 'Distribution of Policies by Province', 'province_distribution.png')
    plot_univariate_categorical(df, 'Gender', 'Distribution of Policies by Gender', 'gender_distribution.png')
    
    # Bivariate analysis
    plot_loss_ratio_by_group(df, 'Province', 'Loss Ratio by Province', 'loss_ratio_province.png')
    plot_loss_ratio_by_group(df, 'Gender', 'Loss Ratio by Gender', 'loss_ratio_gender.png')
    
    # Temporal analysis
    plot_temporal_trends(df)
    
    # Outlier detection
    plot_outliers(df, 'TotalClaims', 'Box Plot of Total Claims', 'total_claims_boxplot.png')
    
    # Creative visualizations
    plot_creative_insights(df, df)
    
    # Statistical tests
    f_stat, p_value = perform_statistical_tests(df, 'Province')
    print(f"\nANOVA for Loss Ratio by Province: F={f_stat:.2f}, p={p_value:.4f}")
    
    return df

if __name__ == "__main__":
    # Example usage
    file_path = 'data/insurance_data.csv'
    main(file_path)
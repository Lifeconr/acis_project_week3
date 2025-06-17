import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

def profile_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Profile dataset with descriptive statistics, data types, and missing values.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.Series]: Numerical stats, data types, missing values.
    """
    numerical_stats = df.select_dtypes(include=['float64', 'int64']).describe()
    data_types = df.dtypes
    missing_values = df.isnull().sum()
    return numerical_stats, data_types, missing_values

def calculate_loss_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Loss Ratio (TotalClaims / TotalPremium).

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with LossRatio column.
    """
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
    df['LossRatio'] = df['LossRatio'].fillna(0)
    return df

def analyze_vehicle_claims(df: pd.DataFrame, top_n: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Identify vehicle makes with highest/lowest claim amounts.

    Args:
        df (pd.DataFrame): Input dataset.
        top_n (int): Number of top/bottom entries to return.

    Returns:
        Tuple[pd.Series, pd.Series]: Top and bottom claims by make.
    """
    if 'make' not in df.columns:
        raise ValueError("Column 'make' not found in DataFrame.")
    claims_by_make = df.groupby('make')['TotalClaims'].sum().sort_values(ascending=False)
    top_claims = claims_by_make.head(top_n)
    bottom_claims = claims_by_make.tail(top_n)
    return top_claims, bottom_claims

def perform_anova(df: pd.DataFrame, group_col: str, target_col: str = 'LossRatio') -> str:
    """
    Perform ANOVA to test differences in target variable across groups.

    Args:
        df (pd.DataFrame): Input dataset.
        group_col (str): Grouping column (e.g., 'Province').
        target_col (str): Target column (default: 'LossRatio').

    Returns:
        str: ANOVA test results summary.
    """
    grouped = df.dropna(subset=[target_col]).groupby(group_col)
    group_data = [group[target_col] for _, group in grouped]
    try:
        f_stat, p_value = f_oneway(*group_data)
        return f"ANOVA for {target_col} by {group_col}: F={f_stat:.2f}, p={p_value:.4f}"
    except Exception as e:
        return f"ANOVA failed for {target_col} by {group_col}: {str(e)}"

def plot_eda_visualizations(df: pd.DataFrame, output_dir: str = '../plots') -> None:
    """
    Generate EDA visualizations including univariate, bivariate, temporal, and creative plots.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        output_dir (str): Directory to save plots (default: '../plots').
    """
    # Univariate: Numerical
    for col, title, filename in [
        ('TotalPremium', 'Distribution of Total Premium', 'total_premium_dist.png'),
        ('TotalClaims', 'Distribution of Total Claims', 'total_claims_dist.png'),
    ]:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], bins=30, kde=True, color='skyblue')
        plt.title(title, fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight')
        plt.show()
        plt.close()

    # Univariate: Categorical
    for col, title, filename in [
        ('Province', 'Policies by Province', 'province_dist.png'),
        ('Gender', 'Policies by Gender', 'gender_dist.png'),
        ('VehicleType', 'Policies by Vehicle Type', 'vehicle_type_dist.png'),
    ]:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, data=df, palette='viridis')
        plt.title(title, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Count', fontsize=12)
        plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight')
        plt.show()
        plt.close()

    # Bivariate: Loss Ratio by Group
    for col, title, filename in [
        ('Province', 'Loss Ratio by Province', 'loss_ratio_province.png'),
        ('Gender', 'Loss Ratio by Gender', 'loss_ratio_gender.png'),
        ('VehicleType', 'Loss Ratio by Vehicle Type', 'loss_ratio_vehicle.png'),
    ]:
        loss_by_group = df.groupby(col)['LossRatio'].mean().sort_values()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=loss_by_group.index, y=loss_by_group.values, palette='magma')
        plt.title(title, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Loss Ratio', fontsize=12)
        plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight')
        plt.show()
        plt.close()

    # Bivariate: Correlation Matrix
    cols = ['TotalPremium', 'TotalClaims', 'LossRatio', 'CustomValueEstimate']
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation of Financial Variables', fontsize=14)
    plt.savefig(f'{output_dir}/correlation_matrix.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Temporal Trends
    claims_over_time = df.groupby('TransactionMonth').agg({
        'TotalClaims': ['count', 'mean']
    }).reset_index()
    claims_over_time.columns = ['TransactionMonth', 'ClaimCount', 'AvgClaimAmount']
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(claims_over_time['TransactionMonth'], claims_over_time['ClaimCount'],
             color='b', label='Claim Frequency')
    ax1.set_xlabel('Transaction Month', fontsize=12)
    ax1.set_ylabel('Claim Count', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(claims_over_time['TransactionMonth'], claims_over_time['AvgClaimAmount'],
             color='r', label='Avg Claim Amount')
    ax2.set_ylabel('Average Claim Amount', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    fig.suptitle('Claim Trends Over Time', fontsize=14)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    plt.savefig(f'{output_dir}/temporal_trends.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Outlier Detection
    for col, title, filename in [
        ('TotalClaims', 'Outliers in Total Claims', 'total_claims_boxplot.png'),
        ('CustomValueEstimate', 'Outliers in Custom Value Estimate', 'custom_value_boxplot.png'),
    ]:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(title, fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.savefig(f'{output_dir}/{filename}', bbox_inches='tight')
        plt.show()
        plt.close()

    # Creative Visualizations
    # 1: Loss Ratio by Province and Gender
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Province', y='LossRatio', hue='Gender', data=df, palette='Set2')
    plt.title('Loss Ratio by Province and Gender', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Loss Ratio', fontsize=12)
    plt.legend(title='Gender')
    plt.savefig(f'{output_dir}/loss_ratio_province_gender.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # 2: Claim Severity by Vehicle Type and Cover Type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='VehicleType', y='TotalClaims', hue='CoverType', data=df, palette='Set3')
    plt.title('Claim Severity by Vehicle Type and Cover Type', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Vehicle Type', fontsize=12)
    plt.ylabel('Total Claims', fontsize=12)
    plt.legend(title='Cover Type')
    plt.savefig(f'{output_dir}/claim_severity_vehicle_cover.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # 3: Heatmap of Claims by Province and Month
    claims_by_month_province = pd.pivot_table(
        df, values='TotalClaims', index='Province', columns='TransactionMonth', aggfunc='sum'
    )
    plt.figure(figsize=(14, 8))
    sns.heatmap(claims_by_month_province, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Total Claims by Province and Month', fontsize=14)
    plt.xlabel('Transaction Month', fontsize=12)
    plt.ylabel('Province', fontsize=12)
    plt.savefig(f'{output_dir}/claims_heatmap.png', bbox_inches='tight')
    plt.show()
    plt.close()
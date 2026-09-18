import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Add src to python path to import our module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import load_data, clean_data, get_preprocessor, save_pipeline_artifacts

def run_eda(df: pd.DataFrame, plots_dir: str = 'plots'):
    """Performs Exploratory Data Analysis and saves key visualizations."""
    os.makedirs(plots_dir, exist_ok=True)
    print("=== Running Exploratory Data Analysis ===")
    
    # 1. Dataset Dimensions & Basics
    print(f"Dataset Shape: {df.shape}")
    print("\n--- Column Types & Non-Null Counts ---")
    print(df.info())
    
    # 2. Missing Values Check (including spaces in TotalCharges)
    total_charges_spaces = (df['TotalCharges'].astype(str).str.strip() == '').sum()
    print("\n--- Missing Values Summary ---")
    print(df.isnull().sum())
    print(f"Empty/whitespace values in 'TotalCharges': {total_charges_spaces}")
    
    # 3. Target Distribution (Churn)
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    print("\n--- Target (Churn) Distribution ---")
    for val in churn_counts.index:
        print(f"Churn '{val}': {churn_counts[val]} ({churn_pct[val]:.2f}%)")
        
    # Generate & Save Plots
    print(f"\nGenerating EDA plots in '{plots_dir}/'...")
    
    # Set seaborn style for premium look
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Plot 1: Target Churn Distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='Churn', data=df, hue='Churn', palette={'Yes': '#E74C3C', 'No': '#2ECC71'}, legend=False)
    plt.title('Distribution of Customer Churn', fontsize=14, pad=15, fontweight='bold')
    plt.xlabel('Churn Status', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() - 300),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', 
                    color='white', fontweight='bold', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'churn_distribution.png'), dpi=150)
    plt.close()
    
    # Plot 2: Numeric Feature Distributions by Churn
    # We first clean TotalCharges locally for visualization
    df_vis = df.copy()
    df_vis['TotalCharges'] = pd.to_numeric(df_vis['TotalCharges'].replace(r'^\s*$', np.nan, regex=True), errors='coerce').fillna(0.0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    titles = ['Tenure (Months)', 'Monthly Charges ($)', 'Total Charges ($)']
    colors = {'Yes': '#E74C3C', 'No': '#3498DB'}
    
    for idx, col in enumerate(numeric_cols):
        sns.kdeplot(data=df_vis, x=col, hue='Churn', fill=True, ax=axes[idx], palette=colors, common_norm=False, alpha=0.4)
        axes[idx].set_title(f'Distribution of {titles[idx]} by Churn', fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_xlabel(titles[idx])
        axes[idx].set_ylabel('Density')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'numeric_distributions.png'), dpi=150)
    plt.close()
    
    # Plot 3: Correlation Heatmap for Numeric Columns
    plt.figure(figsize=(6, 5))
    correlation = df_vis[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True, square=True)
    plt.title('Correlation Heatmap of Numeric Features', fontsize=14, pad=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()

    # Plot 4: Contract Type & Internet Service vs Churn Rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.countplot(data=df, x='Contract', hue='Churn', ax=axes[0], palette={'Yes': '#E74C3C', 'No': '#2ECC71'})
    axes[0].set_title('Churn by Contract Type', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Contract Type')
    axes[0].set_ylabel('Customer Count')
    
    sns.countplot(data=df, x='InternetService', hue='Churn', ax=axes[1], palette={'Yes': '#E74C3C', 'No': '#2ECC71'})
    axes[1].set_title('Churn by Internet Service Type', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Internet Service Type')
    axes[1].set_ylabel('Customer Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'categorical_churn_rates.png'), dpi=150)
    plt.close()
    
    print("EDA completed successfully.")

def main():
    dataset_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # 1. Load raw data
    raw_df = load_data(dataset_path)
    
    # 2. Run Exploratory Data Analysis
    run_eda(raw_df)
    
    # 3. Clean raw data
    cleaned_df = clean_data(raw_df)
    
    # 4. Split into features and target
    X = cleaned_df.drop(columns=['Churn'])
    y = cleaned_df['Churn']
    
    # 5. Define feature groups
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Split into train & test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # 6. Build the preprocessor pipeline
    preprocessor = get_preprocessor(numeric_features, categorical_features)
    
    # 7. Fit and Transform
    print("\nFitting and applying preprocessing pipeline...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get column names after One-Hot Encoding
    # Numeric features retain their names, categorical features expand
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_transformed_cols = numeric_features + encoded_cat_cols
    
    # Convert transformed arrays back to DataFrames for easier viewing and modeling
    X_train_proc_df = pd.DataFrame(X_train_transformed, columns=all_transformed_cols)
    X_test_proc_df = pd.DataFrame(X_test_transformed, columns=all_transformed_cols)
    
    # 8. Save all artifacts
    save_pipeline_artifacts(
        X_train_proc_df, X_test_proc_df, 
        y_train, y_test, 
        preprocessor, 
        output_dir='processed_data'
    )
    
    print("\n=== Pipeline Execution Finished Successfully! ===")

if __name__ == '__main__':
    main()

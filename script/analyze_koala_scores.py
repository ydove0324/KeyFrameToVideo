#!/usr/bin/env python3
"""
Analyze score distributions in Koala-36M dataset
Read CSV files and generate distribution histograms for clarity_score, aesthetic_score, motion_score, video_training_suitability_score
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def analyze_koala_scores(csv_paths, output_path='analysis.png', top_x=30, motion_top_x=50, save_top_csv=False, top_csv_path='top_30_percent_koala.csv'):
    """
    Analyze score data in CSV files and generate distribution histograms
    
    Args:
        csv_paths (list): List of CSV file paths
        output_path (str): Output image path
        top_x (int): Percentage for top analysis for clarity/aesthetic/training_suitability scores (default: 30)
        motion_top_x (int): Percentage for top analysis for motion_score (default: 50)
        save_top_csv (bool): Whether to save filtered data to CSV
        top_csv_path (str): Path for the filtered CSV file
    """
    print(f"Reading CSV files: {csv_paths}")
    
    # Read CSV files, only read needed columns for efficiency
    score_columns = ['clarity_score', 'aesthetic_score', 'motion_score', 'video_training_suitability_score']
    all_columns = ['videoID', 'url', 'timestamp', 'caption', 'clarity_score', 'aesthetic_score', 'motion_score', 'video_training_suitability_score']
    
    # Use chunksize to read large files in batches
    chunk_size = 100000
    all_scores = []
    all_data = []  # Store complete data for CSV export
    
    for csv_path in csv_paths:
        print(f"Processing file: {csv_path}")
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, usecols=all_columns):
            all_scores.append(chunk[score_columns])
            all_data.append(chunk)  # Store complete chunk
            print(f"Read {len(all_scores) * chunk_size} rows of data...")
    
    # Merge all data
    df_scores = pd.concat(all_scores, ignore_index=True)
    df_complete = pd.concat(all_data, ignore_index=True)
    print(f"Total rows read: {len(df_scores)}")
    
    # Check data quality
    print("\nData statistics:")
    print(df_scores.describe())
    
    print("\nMissing values:")
    print(df_scores.isnull().sum())
    
    # Clean data: remove missing values
    df_clean_scores = df_scores.dropna()
    df_clean_complete = df_complete.dropna()
    print(f"\nCleaned data rows: {len(df_clean_scores)}")
    
    # Calculate top X% statistics with different thresholds for motion_score
    print(f"\nTop {top_x}% analysis (motion_score: top {motion_top_x}%):")
    print("=" * 50)
    
    # Calculate percentiles for each metric (motion_score uses different threshold)
    percentiles_threshold = {}
    print(f"\nThreshold values for each metric:")
    print("-" * 50)
    
    for col in score_columns:
        if col == 'motion_score':
            percentile_threshold = (100 - motion_top_x) / 100
            percentiles_threshold[col] = df_clean_scores[col].quantile(percentile_threshold)
            print(f"{col}: {percentiles_threshold[col]:.4f} (top {motion_top_x}% threshold)")
        else:
            percentile_threshold = (100 - top_x) / 100
        percentiles_threshold[col] = df_clean_scores[col].quantile(percentile_threshold)
        print(f"{col}: {percentiles_threshold[col]:.4f} (top {top_x}% threshold)")
    
    # Find samples that are in top X% for all metrics
    top_x_mask = pd.Series([True] * len(df_clean_scores), index=df_clean_scores.index)
    for col in score_columns:
        top_x_mask = top_x_mask & (df_clean_scores[col] >= percentiles_threshold[col])
    
    top_x_count = top_x_mask.sum()
    top_x_percentage = (top_x_count / len(df_clean_scores)) * 100
    
    print(f"\nSamples in top {top_x}% for clarity/aesthetic/training_suitability AND top {motion_top_x}% for motion: {top_x_count}")
    print(f"Percentage of total: {top_x_percentage:.2f}%")
    
    # Save top X% data to CSV if requested
    if save_top_csv:
        top_x_data = df_clean_complete[top_x_mask]
        print(f"\nSaving filtered data to: {top_csv_path}")
        print(f"Number of samples saved: {len(top_x_data)}")
        top_x_data.to_csv(top_csv_path, index=False)
        print(f"Filtered data saved successfully!")
        
        # Print some statistics about the saved data
        print(f"\nStatistics of saved filtered data:")
        print("=" * 50)
        for col in score_columns:
            print(f"{col}:")
            print(f"  Mean: {top_x_data[col].mean():.4f}")
            print(f"  Min: {top_x_data[col].min():.4f}")
            print(f"  Max: {top_x_data[col].max():.4f}")
    
    # Count samples in top X% for each individual metric
    print(f"\nIndividual metric counts:")
    for col in score_columns:
        if col == 'motion_score':
            threshold_percent = motion_top_x
        else:
            threshold_percent = top_x
        top_x_individual = (df_clean_scores[col] >= percentiles_threshold[col]).sum()
        percentage_individual = (top_x_individual / len(df_clean_scores)) * 100
        print(f"{col}: {top_x_individual} samples ({percentage_individual:.2f}% - top {threshold_percent}%)")
    
    # Set font for English labels
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Koala-36M Dataset Score Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Define colors and titles
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    titles = ['Clarity Score Distribution', 'Aesthetic Score Distribution', 'Motion Score Distribution', 'Training Suitability Score Distribution']
    
    for i, (col, color, title) in enumerate(zip(score_columns, colors, titles)):
        row = i // 2
        col_idx = i % 2
        ax = axes[row, col_idx]
        
        # Draw histogram
        ax.hist(df_clean_scores[col], bins=50, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Score Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistical information
        mean_val = df_clean_scores[col].mean()
        std_val = df_clean_scores[col].std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.legend()
        
        # Add text statistical information
        stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {df_clean_scores[col].min():.3f}\nMax: {df_clean_scores[col].max():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis chart saved to: {output_path}")
    
    # Print detailed statistical information
    print("\nDetailed statistics:")
    print("=" * 50)
    for col in score_columns:
        print(f"\n{col}:")
        print(f"  Mean: {df_clean_scores[col].mean():.4f}")
        print(f"  Median: {df_clean_scores[col].median():.4f}")
        print(f"  Std: {df_clean_scores[col].std():.4f}")
        print(f"  Min: {df_clean_scores[col].min():.4f}")
        print(f"  Max: {df_clean_scores[col].max():.4f}")
        print(f"  25th percentile: {df_clean_scores[col].quantile(0.25):.4f}")
        print(f"  75th percentile: {df_clean_scores[col].quantile(0.75):.4f}")

if __name__ == "__main__":
    # CSV file paths
    csv_paths = [
        "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_1.csv",
        "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_2.csv",
        "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_3.csv",
        "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_4.csv",
        "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_5.csv",
        "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_6.csv",
        "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_7.csv",
        "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_8.csv",
        "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_9.csv",
        "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_10.csv",
    ]
    
    # Check if files exist
    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            print(f"Error: File does not exist: {csv_path}")
            exit(1)
    
    # Execute analysis
    analyze_koala_scores(csv_paths, 'analysis.png', top_x=30, motion_top_x=50, save_top_csv=True, top_csv_path='top_30_percent_koala.csv') 
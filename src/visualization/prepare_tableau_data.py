import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def prepare_engagement_metrics(df):
    """Prepare customer engagement metrics for Tableau."""
    engagement_metrics = df.groupby('date').agg({
        'clicks': 'sum',
        'time_spent': 'mean',
        'page_views': 'sum',
        'purchase_conversion': 'mean'
    }).reset_index()
    
    # Calculate rolling averages
    engagement_metrics['rolling_7d_conversion'] = engagement_metrics['purchase_conversion'].rolling(7).mean()
    engagement_metrics['rolling_7d_clicks'] = engagement_metrics['clicks'].rolling(7).mean()
    
    return engagement_metrics

def prepare_ad_metrics(df):
    """Prepare advertising metrics for Tableau."""
    ad_metrics = df.groupby(['campaign_id', 'date']).agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'cost': 'sum',
        'purchase_conversion': 'mean'
    }).reset_index()
    
    # Calculate derived metrics
    ad_metrics['ctr'] = ad_metrics['clicks'] / ad_metrics['impressions']
    ad_metrics['cpc'] = ad_metrics['cost'] / ad_metrics['clicks']
    ad_metrics['conversion_rate'] = ad_metrics['purchase_conversion']
    
    return ad_metrics

def prepare_customer_segments(df):
    """Prepare customer segment analysis for Tableau."""
    customer_segments = df.groupby('customer_segment').agg({
        'purchase_conversion': 'mean',
        'time_spent': 'mean',
        'clicks': 'mean',
        'cost': 'sum'
    }).reset_index()
    
    return customer_segments

def prepare_model_predictions(df, predictions):
    """Prepare model predictions for Tableau."""
    prediction_data = pd.DataFrame({
        'customer_id': df['customer_id'],
        'actual_conversion': df['purchase_conversion'],
        'predicted_probability': predictions,
        'date': df['date']
    })
    
    return prediction_data

def main():
    # Load the processed data
    data_path = 'data/processed/training_data.csv'
    df = pd.read_csv(data_path)
    
    # Create output directory
    output_dir = 'data/tableau'
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare different metric views
    engagement_metrics = prepare_engagement_metrics(df)
    ad_metrics = prepare_ad_metrics(df)
    customer_segments = prepare_customer_segments(df)
    
    # Save the prepared data
    engagement_metrics.to_csv(f'{output_dir}/engagement_metrics.csv', index=False)
    ad_metrics.to_csv(f'{output_dir}/ad_metrics.csv', index=False)
    customer_segments.to_csv(f'{output_dir}/customer_segments.csv', index=False)
    
    print("Tableau data preparation completed!")
    print(f"Files saved in: {output_dir}")

if __name__ == "__main__":
    main() 
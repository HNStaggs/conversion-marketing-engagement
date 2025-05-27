# Customer Engagement and Ad Performance Prediction

This project implements a machine learning solution to predict purchase conversion based on customer engagement and advertising metrics. The solution is deployed on AWS SageMaker and includes a Tableau dashboard for visualization.

## Project Overview

- **Objective**: Predict purchase conversion probability using customer engagement and advertising metrics
- **Model Type**: Binary Classification
- **Deployment**: AWS SageMaker
- **Visualization**: Tableau Public Dashboard

## Project Structure

```
├── data/                      # Data storage
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── notebooks/                # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
├── src/                      # Source code
│   ├── data/                # Data processing scripts
│   ├── features/            # Feature engineering scripts
│   ├── models/              # Model training and evaluation
│   └── visualization/       # Tableau dashboard preparation
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials:
```bash
aws configure
```

## Data Requirements

The model requires the following types of data:
- Customer engagement metrics (clicks, time spent, page views)
- Advertising metrics (impressions, CTR)
- Purchase conversion data
- Customer demographics (gender)

## Model Training

1. Data preprocessing and feature engineering
2. Model training using AWS SageMaker
3. Model evaluation and validation
4. Model deployment

## Tableau Dashboard

The Tableau dashboard includes:
- Purchase conversion rates
- Customer engagement metrics
- Advertising performance metrics
- Model prediction insights
- Interactive visualizations

## AWS SageMaker Integration

The project uses AWS SageMaker for:
- Model training and deployment
- Batch inference
- Real-time predictions
- Model monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
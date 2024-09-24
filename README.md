# Time-Series LSTM Neural Network

This repository focuses on time series analysis using a Long Short-Term Memory (LSTM) Neural Network applied to financial data. The project aims to predict future price movements using deep learning techniques and evaluate the strategy's performance on financial data.

## Project Overview

Time series analysis plays a critical role in forecasting financial data, especially for trading and investment strategies. This project uses historical financial data, applies preprocessing techniques, and employs an LSTM neural network to predict price movements for forex data. LSTMs are ideal for sequential data as they can capture temporal dependencies and trends over time. 

### Key Features

- **Data Retrieval**: Custom functions to fetch historical financial data.
- **Data Processing**: Includes creating shifted columns and transforming data into a 3D structure for LSTM models.
- **LSTM Model Development**: The LSTM neural network is built using TensorFlow and Keras, consisting of:
  - Input layer reshaped for 3D time series data
  - 2 LSTM layers with ReLU activation and Dropout layers for regularization
  - Output layer to predict price direction
  - **EarlyStopping**: To monitor validation loss and stop training when overfitting is detected
  - **ModelCheckpoint**: Saves the best-performing model based on validation loss.
- **Backtesting and Performance Evaluation**: Financial metrics such as Sortino, Beta, and Alpha ratios are calculated. Drawdowns and cumulative returns are visualized.
- **Visualization**: Includes functions to plot cumulative returns, test results, and drawdowns.

## Repository Structure

- **`data.py`**: Contains functions for retrieving and manipulating financial data, including:
  - `get_rates`: Fetches historical price data for a given symbol and timeframe.
  - `add_shifted_columns`: Adds lagged features to the DataFrame for time series predictions.
  - `split_data`: Splits the data into training, validation, and test sets.
  - `create_3d_data`: Reshapes data into 3D format suitable for LSTM input.

- **`backtest.py`**: Provides functions for backtesting financial strategies, including:
  - `compute_strategy_returns`: Calculates cumulative returns for a given strategy.
  - `compute_drawdown`: Computes the drawdown for a strategy.
  - `compute_model_accuracy`: Computes the accuracy of model predictions.
  - `vectorize_backtest_returns`: Computes financial metrics such as Sortino, Beta, and Alpha ratios.
  - `create_model_checkpoint`: Saves the model during training based on performance metrics.

- **`ai.py`**: Contains the primary LSTM model training function:
  - `run_lstm_1`: Trains the LSTM model using the processed data.

- **`display.py`**: Contains functions for visualizing results:
  - `plot_test_returns`: Plots cumulative test returns for the model.
  - `plot_drawdown`: Plots the drawdown of the model's performance.

- **`LSTM Neural Network.ipynb`**: Jupyter Notebook containing the code for building, training, and evaluating the LSTM Neural Network. The notebook walks through the following steps:
  - Data processing
  - Model creation and compilation using TensorFlow and Keras
  - Training the model with EarlyStopping and ModelCheckpoint callbacks
  - Evaluating the model on the validation set
  - Backtesting model performance on financial data

- **`.gitignore`**: Specifies files and directories to be ignored by Git.

- **`LICENSE`**: The MIT License under which this project is distributed.

- **`README.md`**: This file.

## Installation

To run this project, you need Python installed with the following packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tensorflow`
- `ipywidgets`

You can install the necessary packages by running:


## How to View

**Clone the Repository for Viewing**: You may clone this repository to your local machine for personal review and educational purposes only:
   ```bash 
   git clone https://github.com/YarosInter/Time-Series-LSTM-Neural-Network.git
   ```
   
   
## Contributing

This repository is for personal or educational purposes only. Contributions and modifications are not permitted unless explicitly allowed. Feel free to reach out if you'd like to collaborate or contribute, let’s learn together!


## Disclaimer

The code in this repository is for educational or personal review only and is not licensed for use, modification, or distribution. 
This code is part of my journey in learning and experimenting with new ideas. It’s shared for educational purposes and personal review. Please feel free to explore, but kindly reach out for permission if you’d like to use, modify, or distribute any part of it.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from datetime import datetime
from UsefulFunctions import data
import cufflinks as cf
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from tensorflow.keras.layers import Dropout



def compute_strategy_returns(y_test, y_pred):
    """
    Computes the strategy returns by comparing real percent changes with model predictions.

    This function creates a DataFrame that includes the actual percent changes (`y_test`) and the model's
    predicted values (`y_pred`). It then calculates the directional positions based on the actual and predicted
    values, and computes the returns by multiplying the real percent change by the predicted position.

    Args:
        y_test (Series or DataFrame): The actual percent changes (real values).
        y_pred (Series or DataFrame): The predicted values from the model.

    Returns:
        DataFrame: A DataFrame containing the actual percent changes, predicted values, 
                   real positions, predicted positions, and computed returns.
    """

    # Initialize a DataFrame with the actual percent changes and add the model's predictions
    df = pd.DataFrame(y_test)
    df["prediction"] = y_pred

    # Add columns for the real and predicted directional positions
    df["real_position"] = np.sign(df["pct_change"])
    df["pred_position"] = np.sign(df["prediction"])

    # Calculate the strategy returns by multiplying the actual percent change by the predicted position
    # Note: Predictions are based on the previous bar's data, so no additional shift is needed
    df["returns"] = df["pct_change"] * df["pred_position"]

    return df



def compute_model_accuracy_cufflinks(real_positions, predicted_positions, name=" "):
    """
    Computes and displays the accuracy of predicted positions compared to real positions.

    Parameters:
    real_positions (list or array-like): The actual positions.
    predicted_positions (list or array-like): The positions predicted by the model.

    Returns:
    pd.DataFrame: A DataFrame containing the real positions, predicted positions, and accuracy (1 for correct, 0 for incorrect).
    
    Displays:
    - Counts of correct and incorrect predictions.
    - Histogram showing the distribution of accuracy values.
    - Model accuracy percentage.
    """

    # Initialize Cufflinks in offline mode
    cf.go_offline()

    # Creating Dataframe with real positions and predicted positions
    df_accuracy = pd.DataFrame(real_positions, columns=["real_position"])
    df_accuracy["pred_position"] = predicted_positions
    
    # Assigning 1 if the position forecasted is equal to the real position and 0 otherwise
    df_accuracy["accuracy"] = np.where(df_accuracy["real_position"] == df_accuracy["pred_position"], 1, 0)

    # Count the occurrences of each unique accuracy value in the 'accuracy' column and store the result in 'accuracy'
    accuracy = df_accuracy["accuracy"].value_counts()

    # Printing explanation for the counts of 0 and 1 in the 'accuracy' column
    print("Counts of 0 indicate instances where the predicted position did not match the real position.")
    print("Counts of 1 indicate instances where the predicted position matched the real position.\n")
    print(accuracy)

    # Total counts of occurrences where the model was right (number assigned 1) divided by the total number of predictions
    model_accuracy = accuracy[1] / len(df_accuracy)
    print(f"\nModel has an accuracy of: {model_accuracy * 100:.2f}%")
    
    # Plotting the accuracy of the model in a histogram using the dynamic plot with Cufflinks
    df_accuracy["accuracy"].iplot(
        kind="hist",       
        xTitle="Prediction Result", 
        yTitle="Counts",
        title=f"Model Accuracy {name}",
        bargap=0.2,
        theme="white",         
        colors=["blue"],
        #layout=dict(height=400)
    )
        
    return df_accuracy.head()



def compute_model_accuracy(real_positions, predicted_positions, name=" "):
    """
    Computes and displays the accuracy of predicted positions compared to real positions.

    Parameters:
    real_positions (list or array-like): The actual positions.
    predicted_positions (list or array-like): The positions predicted by the model.

    Returns:
    pd.DataFrame: A DataFrame containing the real positions, predicted positions, and accuracy (1 for correct, 0 for incorrect).
    
    Displays:
    - Counts of correct and incorrect predictions.
    - Bar plot showing the distribution of accuracy values with a gap between bars.
    - Model accuracy percentage.
    """
    
    # Creating DataFrame with real positions and predicted positions
    df_accuracy = pd.DataFrame({'real_position': real_positions})
    df_accuracy["pred_position"] = predicted_positions
    
    # Assigning 1 if the position forecasted is equal to the real position and 0 otherwise
    df_accuracy["accuracy"] = np.where(df_accuracy["real_position"] == df_accuracy["pred_position"], 1, 0)

    # Count the occurrences of each unique accuracy value in the 'accuracy' column
    accuracy = df_accuracy["accuracy"].value_counts()

    # Printing explanation for the counts of 0 and 1 in the 'accuracy' column
    print("Counts of 0 indicate instances where the predicted position did not match the real position.")
    print("Counts of 1 indicate instances where the predicted position matched the real position.\n")
    print(accuracy)

    # Total counts of occurrences where model was right (number assigned 1) divided into the total number of predictions
    model_accuracy = accuracy[1] / len(df_accuracy)
    print(f"\nModel has an accuracy of: {model_accuracy * 100:.2f}%")

    # Create a bar plot with a gap between bars
    plt.bar(accuracy.index, accuracy.values, width=0.8)  # width adjusted for bar gap
    plt.xticks([0, 1], labels=['Incorrect = 0', 'Correct = 1'])
    plt.title("Model Accuracy", fontsize=20)
    plt.ylabel("Counts", fontsize=15)
    plt.xlabel(f"Prediction Result {name}", fontsize=15)
    plt.show()

    return df_accuracy.head()

    

def vectorize_backtest_returns(returns, anualization_factor, benchmark_asset=".US500Cash", mt5_timeframe=mt5.TIMEFRAME_H1):
    """
    Computes and prints the Sortino Ratio, Beta Ratio, and Alpha Ratio for a given set of returns series.
    
    Parameters:
    - returns: Series of returns from a strategy.
    - anualization_factor: Factor used to annualize returns.
    - benchmark_asset: The benchmark asset for comparison (default is S&P 500).
    - mt5_timeframe: Timeframe for pulling benchmark data (default is 1H).

    Note: The timeframe for benchmark data must match the timeframe of the strategy's returns for accurate results.
    
    Returns:
    None
    """

    ### Computing Sortino Ratio ###

    # Sortino Ratio is being calculated without a Risk-Free Rate
    mean_return = np.mean(returns)
    downside_deviation = np.std(returns[returns < 0])
    
    # Number of 15-minute periods in a year
    periods_per_year = anualization_factor * 252
    
    # Annualizing the mean return and downside deviation
    annualized_mean_return = mean_return * periods_per_year
    annualized_downside_deviation = downside_deviation * np.sqrt(periods_per_year)
    
    # Calculating the annualized Sortino ratio
    annualized_sortino = annualized_mean_return / annualized_downside_deviation
    
    print(f"Sortino Ratio: {'%.3f' % annualized_sortino}")
    if annualized_sortino > 0:
        print("- Positive Sortino (> 0): The investment’s returns exceed the target return after accounting for downside risk.\n")
    else:
        print("- Negative Sortino (< 0): The investment’s returns are less than the target return when considering downside risk.\n")

    ### Computing Beta Ratio ###

    print("***Asset for Benchamark is S&P500***\n")

    # Fetching the oldest date from the X_test set to pull data from the same date for the S&P 500
    date = returns.index.min()
    
    # Extracting the year, month, and day from the date
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    min = date.minute
    sec = date.second
    
    # Pulling S&P 500 data from the specified date and time
    sp500_data = data.get_rates(".US500Cash", mt5.TIMEFRAME_H1, from_date=datetime(year, month, day))
    sp500_data = sp500_data[["close"]]
    
    # Computing the returns on the S&P 500
    sp500_data["returns"] = sp500_data["close"].pct_change(1)
    sp500_data.drop("close", axis=1, inplace=True)
    sp500_data.dropna(inplace=True)
    
    # Concatenate values between the returns in the predictions and the returns in the S&P 500
    val = pd.concat((returns, sp500_data["returns"]), axis=1)
    
    # Changing column names to identify each one
    val.columns.values[0] = "Returns Pred"
    val.columns.values[1] = "Returns SP500"
    val.dropna(inplace=True)

    # Calculating Beta Ratio
    covariance_matrix = np.cov(val.values, rowvar=False)
    covariance = covariance_matrix[0][1]
    variance = covariance_matrix[1][1]
    beta = covariance / variance
    
    print(f"Beta Ratio: {'%.3f' % beta}")
    if beta == 1:
        print("- Beta ≈ 1: The asset moves in line with the market.\n")
    elif beta < 1:
        print("- Beta < 1: The asset is less volatile than the market (considered less risky).\n")
    else:
        print("- Beta > 1: The asset is more volatile than the market (higher potential return but also higher risk).\n")

    ### Computing Alpha Ratio ###

    alpha = (anualization_factor * 252 * mean_return * (1 - beta)) * 100
    
    print(f"Alpha Ratio: {'%.3f' % alpha}")
    if alpha > 0:
        print("- Positive Alpha (> 0): The investment outperformed the market.")
    else:
        print("- Negative Alpha (< 0): The investment underperformed the market.")



def compute_drawdown(returns):
    """
    Computes the drawdown of a given series of returns.

    Drawdown represents the decline from a historical peak in the cumulative return of an investment.

    Parameters:
    -----------
    returns : pandas.Series or numpy.ndarray
        A series of periodic returns (typically in percentage change or relative return format).

    Returns:
    --------
    pandas.Series or numpy.ndarray
        A series representing the drawdown, which is the percentage drop from the running maximum of the cumulative return.   
    """
    
    # Compute cumulative return
    cumulative_return = returns.dropna().cumsum() + 1

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_return)

    # Computing the drawdown
    drawdown = cumulative_return / running_max - 1
    return drawdown



def create_model_checkpoint(model_name, save_path="model_experiments"):
    """
    Creates a ModelCheckpoint callback to save the best-performing version of a model during training.
    
    Args:
        model_name (str): The name of the model to be used for saving the file.
        save_path (str, optional): The directory path where the model file will be saved. Defaults to "model_experiments".
    
    Returns:
        ModelCheckpoint: A callback that saves the model with the lowest validation loss.
    
    Notes:
        - The checkpoint saves the model in the provided directory (`save_path`) with the format "{model_name}.keras".
        - Only the model with the best validation loss is saved.
    """
    # Ensure the directory exists
    #os.makedirs(save_path, exist_ok=True)
    
    return ModelCheckpoint(
        filepath=os.path.join(save_path, f"{model_name}.keras"),
        monitor="val_loss",
        verbose=0,
        save_best_only=True
    )


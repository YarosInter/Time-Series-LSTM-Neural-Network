import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from UsefulFunctions import data
import cufflinks as cf
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.graph_objs as go



def plot_test_returns_cufflinks(returns, name=" "):
    """
    Plots the cumulative returns of one or multiple test sets along with a break-even line.

    This function takes a 2D array-like object `returns` and converts it into a pandas DataFrame, 
    or takes an already pandas DataFrame with multiple columns of returns, one for each test,
    plots the cumulative returns for each column as a separate line plot. It also includes 
    a horizontal break-even line at y = 0 to indicate no profit or loss.

    Parameters:
    -----------
    returns : array-like
        A 2D array-like object (such as a list of lists or a pandas DataFrame) where each column 
        represents the returns of a different test set over time.

    name : string, optional
        A string to be used in the plot title to describe the dataset or the specific test set. 
        Default is an empty string, which will result in the title "Cumulative Returns" without 
        additional description.

    Returns:
    --------
    None
        The function displays the plot using Plotly but does not return any value.
    """

    # Initialize Plotly offline mode
    pyo.init_notebook_mode(connected=True)

    df = pd.DataFrame(returns)

    # Create a subplot
    fig = make_subplots(rows=1, cols=1)

    # Add cumulative returns data for test set
    for i in range(len(df.columns)):
        trace_test = go.Scatter(x=df.index, y=df.iloc[:, i].cumsum() * 100, mode="lines", name=f"Retruns Test {i}")
        fig.add_trace(trace_test)
    
    # Add break-even line
    trace_break_even = go.Scatter(x=df.index, y=[0] * len(df.index), mode="lines", name="Break-even", line=dict(color="red"))
    fig.add_trace(trace_break_even)

    # Set layout
    fig.update_layout(title=f"Cumulative Returns {name}", xaxis_title="Time", yaxis_title="P&L in %", height=450)

    # Display the plot
    pyo.iplot(fig)
    print(f"Profits : {'%.2f' % (df.cumsum().iloc[-1].sum() * 100)}%")



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



def plot_drawdown_cufflinks(return_series):
    """
    Computes and visualizes the drawdown of a strategy based on its return series.

    Parameters:
    return_series (pd.Series): A pandas Series containing the return series of the strategy. 
                               Each value represents the return for a specific period.

    Displays:
    - A plot showing the drawdown over time as a filled area chart.
    - The maximum drawdown percentage is printed to the console.

    Notes:
    - The function assumes the return series is cumulative and starts at zero.
    - NaN values in the return series are dropped before computation.
    - If the return series is empty or contains only NaN values, no plot will be generated.
    """
    
    if return_series.dropna().empty:
        print("The return series is empty or contains only NaN values.")
        return

    # Compute cumulative return
    cumulative_return = return_series.dropna().cumsum() + 1

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_return)

    # Computing the drawdown
    drawdown = cumulative_return / running_max - 1

    fig = make_subplots(rows=1, cols=1)

    trace_test = go.Scatter(x=drawdown.index, 
                            y=drawdown * 100, 
                            mode="lines", 
                            fill="tozeroy", 
                            name="Drawdown", 
                            line=dict(color="red"),
                            fillcolor="rgba(255, 0, 0, 0.8)") # Red color with 0.8 opacity
                            
    fig.add_trace(trace_test)
    
    fig.update_layout(title="Strategy Drawdown", xaxis_title="Time", yaxis_title="Drawdown in %", height=450, showlegend=True)
    pyo.iplot(fig)

    maximum_drawdown = np.min(drawdown) * 100
    print(f"Max Drawdown: {'%.2f' % maximum_drawdown}%")



def plot_drawdown(return_series, name=" "):
    """
    Computes and visualizes the drawdown of a strategy based on its return series.

    Parameters:
    return_series (pd.Series): A pandas Series containing the return series of the strategy. 
                               Each value represents the return for a specific period.

    Displays:
    - A plot showing the drawdown over time as a filled area chart.
    - The maximum drawdown percentage is printed to the console.

    Notes:
    - The function assumes the return series is cumulative and starts at zero.
    - NaN values in the return series are dropped before computation.
    - If the return series is empty or contains only NaN values, no plot will be generated.
    """
    
    if return_series.dropna().empty:
        print("The return series is empty or contains only NaN values.")
        return

    # Compute cumulative return
    cumulative_return = return_series.dropna().cumsum() + 1

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_return)

    # Computing the drawdown
    drawdown = cumulative_return / running_max - 1

    plt.figure(figsize=(15, 4))
    plt.fill_between(drawdown.index, drawdown * 100, 0, drawdown, color="red", alpha=0.70)
    plt.title(f"Strategy Drawdown {name}", fontsize=20)
    plt.ylabel("Drawdown %", fontsize=15)
    plt.xlabel("Time")
    plt.show()

    maximum_drawdown = np.min(drawdown) * 100
    print(f"Max Drawdown: {'%.2f' % maximum_drawdown}%")



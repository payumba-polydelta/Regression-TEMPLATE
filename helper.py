import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display_html, display_markdown, HTML, Markdown as md
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.lines import Line2D
import math
import re
from scipy import stats
import pickle
from joblib import dump
import time
from typing import Union

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, GridSearchCV


NUM_DECIMAL_PLACES = 7

def load_data(file_name: str,
              dropped_columns: list[str],
              na_value_representations: list[str]) -> pd.DataFrame:
    """
    Loads in user's input file as a pandas DataFrame and converts various representations of missing
    values to NaN. The file should be stored in the 'data' directory.
    
    Args:
        file_name (str): Name of file containing data for clustering
        dropped_columns (list[str]): List of columns to drop from the dataframe
        na_value_representations (list[str]): List of strings that represent missing values in
            the dataset
    Returns:
        df (pd.DataFrame): Dataframe of variable values for all data entries
    """
    # Automatically prepends 'data/' to the file name
    file_name: str = "data/" + file_name
    file_extension: str = file_name.split(".")[-1]

    if file_extension == "csv":
        df = pd.read_csv(file_name)
    elif file_extension in ["xls", "xlsx"]:
        if file_extension == "xls":
            df = pd.read_excel(file_name, engine = 'xlrd')
        else:
            df = pd.read_excel(file_name, engine = 'openpyxl')
    elif file_extension == "json":
        df = pd.read_json(file_name)
    else:
        raise ValueError("""Unsupported file format or misspelled file name. Please upload 
                         a CSV, Excel, or JSON file and ensure the file name is spelled correctly.""")
    
    # Replaces input representations of missing values with np.nan
    df = df.replace(na_value_representations, np.nan)
    
    df = df.drop_duplicates()
    df = df.drop(columns = dropped_columns)
    
    return df


def get_number_of_unique_categories(df: pd.DataFrame, categorical_columns: list[str]) -> dict[str, int]:
    """
    Calculate the number of unique classes for each categorical column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        categorical_columns (list[str]): A list of column names representing categorical variables.

    Returns:
        dict[str, int]: A dictionary where keys are categorical column names and values are the number of unique classes.
    """
    num_unique_dict = {}
    
    display_text("Number of Unique Categories in Categorical Columns:", font_size = 20)
    for column in categorical_columns:
        if column in df.columns:
            num_unique_dict[column] = df[column].nunique()
            display_text(f"Number of Unique Categories in '{column}': {num_unique_dict[column]}")
        else:
            display_text(f"Warning: Column '{column}' not found in the DataFrame.")
    
    print()
    return num_unique_dict


def display_df(df: pd.DataFrame,
                      font_size: int = 14) -> None:
    """
    Displays the passed in DataFrame with the specified font size.

    Args:
        df (pd.DataFrame): The DataFrame to be displayed.
        font_size (int): The font size at which the items in the
        DataFrame should be displayed.

    Returns:
        None
    """
    df_html = df.to_html()
    styled_html = f'<div style="font-size: {font_size}px;">{df_html}</div>'
    display_html(HTML(styled_html))
    

def display_text(text: str,
                 font_size: int = 16,
                 font_weight = 'normal') -> None:
    """
    Displays the passed in text with the specified font size and font weight.

    Args:
        text (str): The text to be displayed.
        font_size (int): The font size at which the text should be displayed.
        font_weight: The font weight (e.g., 'normal', 'bold', 'bolder', 'lighter',
        or numeric value from 100 to 900).

    Returns:
        None
    """
    styled_html = f'<div style="font-size: {font_size}px; font-weight: {font_weight};">{text}</div>'
    display_html(HTML(styled_html))
    
    
def string_to_float(value_str: str):
    """
    Cleans a string by removing all characters except digits and decimal points directly followed by a digit
    using regular expressions and converts the cleaned string to a float. This function may not work as
    intended with some strings that contain multiple and/or awkwardly placed decimal points.

    Args:
        value_str (str): The string to be cleaned and converted.

    Returns:
        float or None: The cleaned float value or None if conversion fails.
    """
    # Check if the value is np.nan and returns None if it is
    if pd.isna(value_str):
        return None
    
    # Remove all characters except digits and decimal points followed by a digit.
    cleaned_str: str = re.sub(r'[^0-9.]+', '', value_str)
    
    # Additional check to handle multiple decimal points or trailing decimal points
    parts: list[str] = cleaned_str.split('.')
    if len(parts) > 2:
        cleaned_str: str = ''.join(parts[:-1]) + '.' + parts[-1]
    elif len(parts) == 2 and parts[1] == '':
        cleaned_str: str = parts[0]
    
    # Attempt to convert string to float
    try:
        float_value = pd.to_numeric(cleaned_str)
    except ValueError:
        print(f"Failed to convert {value_str} to a float")
        # Sets the float_value to None if the string cannot be converted to a float
        float_value = None
    
    return float_value


def drop_rows_with_missing_values(df: pd.DataFrame,
                                  columns_to_check: list[str]) -> pd.DataFrame:
    """
    Makes a copy of the input DataFrame and drops rows that have one or more missing values in any of the columns specified by 
    the columns_to_check parameter (does not mutate the input DataFrame). Also prints the number of entries dropped and the
    resulting total number of entries.
    
    Args:
        df (pd.DataFrame): DataFrame containing loded in data
        columns_to_check (list[str]): List of columns to check for missing values
    Returns:
        dropna_df (pd.DataFrame): DataFrame with missing values dropped
    """
    
    original_number_of_entries = len(df)
    
    dropna_df = df.dropna(subset = columns_to_check)
    new_number_of_entries = len(dropna_df)
    number_of_entries_dropped = original_number_of_entries - new_number_of_entries
    
    display_text(f"drop_rows_with_missing_values Results: {number_of_entries_dropped} Entries Dropped") 
    display_text(f"New Number of Entries: {new_number_of_entries}")
    
    return dropna_df


def impute_missing_values(df: pd.DataFrame,
                          numerical_columns_to_impute: list[str],
                          categorical_columns_to_impute: list[str]) -> pd.DataFrame:
    """
    Imputes missing values in the DataFrame with either the median value (for numerical variables) or the most frequent value
    (for categorical variables).
    
    Args:
        numerical_columns_to_impute (list[str]): List of the names of numerical columns with missing values to impute
        categorical_columns_to_impute (list[str]): List of the names of categorical columns with missing values to impute
    Returns:
        impute_df (pd.DataFrame): DataFrame with missing values imputed
    """
    impute_df = df.copy()
    
    # Here is where to configure the imputation strategy if need be
    numerical_imputer = SimpleImputer(strategy = "median")
    categorical_imputer = SimpleImputer(strategy = "most_frequent")
    
    impute_df[numerical_columns_to_impute] = numerical_imputer.fit_transform(impute_df[numerical_columns_to_impute])
    impute_df[categorical_columns_to_impute] = categorical_imputer.fit_transform(impute_df[categorical_columns_to_impute])
    
    display_text("Missing Values Successfully Imputed")
    
    return impute_df


def visualize_outliers(df: pd.DataFrame,
                       numerical_columns_to_check: Union[list[str], str],
                       iqr_multiplier: float = 1.5,
                       display: bool = True,
                       remove: bool = False,
                       remove_option: str = "both") -> pd.DataFrame:
    """
    Creates a boxplot for each column of the input Dataframe in the numerical_columns_to_check parameter to help users visualize potential
    outliers in their dataset. Below this boxplot, the function prints the number of high and low outliers (determined by the IQR method) in the
    current column. The upper and lower bounds for outliers are denoted by red dotted lines. Points below the low bound red dotted line
    or above the high bound red dotted line are consideered outliers. Users can choose whether to drop outlier entries through the remove
    boolean parameter. You can change which points are considered outliers by changing the iqr_multiplier parameter.
    
    The lower and upper whiskers of the boxplot denote the 5th and 95th percentile of the current column's values respectively.

    This function can be used iteratively to handle outliers in different columns with varying sensitivity levels. It allows for
    selective removal of entries below/above the red dotted lines. The function can be run without displaying visualizations for efficiency.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be analyzed.
        numerical_columns_to_check (Union[list[str], str], default = numerical_variables): List of the names of columns to check for outliers. The
            default argument is a list of all numerical columns in the input DataFrame.
        iqr_multiplier (float, default = 1.5): Multiplier for the IQR to define the outlier threshold. Higher values are more lenient, increasing
            the range of the upper and lower red dotted lines. Lower values are more strict, decreasing the range of the red dotted lines.
        display (bool, default = True): If True, displays boxplots for each variable. If false, only outlier statistics are printed.
        remove (bool, default = False): If True, removes identified outliers from the DataFrame.
        remove_option (str, default = 'both'): Specifies which outliers to remove: 'both' removes all identified outliers, 'upper' only removes
            outliers greater than the upper bound (values past the upper red dotted line), and 'lower' only removes outliers less than the
            lower bound (values behind the lower red dotted line). This parameter has no effect if remove = False.

    Returns:
        pd.DataFrame: The original DataFrame if remove = False, otherwise a new DataFrame with outliers removed.
    """
    # If a single column is passed in as a string, convert it to a list so the following for loop still works properly
    if type(numerical_columns_to_check) == str:
        numerical_columns_to_check = [numerical_columns_to_check]
        
    for col in numerical_columns_to_check:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (iqr * iqr_multiplier)
        upper_bound = q3 + (iqr * iqr_multiplier)
        
        # Only create plot if display is True
        if display:
            plt.figure(figsize = (10, 6))
            ax = sns.boxplot(x = df[col], whis = [5, 95])
            plt.title(f'Boxplot of {col}')
            
            # Add vertical red dotted lines for lower and upper bounds if within the plot's x-axis limits
            x_min, x_max = ax.get_xlim()
            if x_min <= lower_bound <= x_max:
                plt.axvline(lower_bound, color='red', linestyle='dotted', linewidth=1)
            if x_min <= upper_bound <= x_max:
                plt.axvline(upper_bound, color='red', linestyle='dotted', linewidth=1)
            
            # Create legend
            legend_lines = [Line2D([0], [0], color='red', linestyle='dotted', linewidth=1)]
            legend_labels = ['Lower/Upper Bound']
            plt.legend(legend_lines, legend_labels, loc='upper right')
            plt.show()
        
        lower_outlier_count = df[col][df[col] < lower_bound].count()
        upper_outlier_count = df[col][df[col] > upper_bound].count()
        
        display_text(f"{col}:", font_size = 18, font_weight = 'bold')
        display_text(f"- Lower Bound for Outliers: {lower_bound}", font_size = 16)
        display_text(f"- Upper Bound for Outliers: {upper_bound}", font_size = 16)
        display_text(f"- Number of Outliers Below Lower Bound: {lower_outlier_count}", font_size = 16)
        display_text(f"- Number of Outliers Above Upper Bound: {upper_outlier_count}", font_size = 16)
        print()
        
    # Removes outliers from the DataFrame if remove = True
    if remove:
        # Calculate indices of outliers
        lower_outlier_indices = df.index[df[col] < lower_bound].tolist()
        upper_outlier_indices = df.index[df[col] > upper_bound].tolist()
        outlier_indices_to_be_removed = set()
        
        # Add outlier indices that will be removed to outlier_indices_to_be_removed based on the remove_option parameter
        if remove_option == "both":
            outlier_indices_to_be_removed.update(lower_outlier_indices)
            outlier_indices_to_be_removed.update(upper_outlier_indices)
        elif remove_option == "lower":
            outlier_indices_to_be_removed.update(lower_outlier_indices)
        elif remove_option == "upper":
            outlier_indices_to_be_removed.update(upper_outlier_indices)
        else:
            raise ValueError("Invalid argument passed into remove_option parameter. Please use 'both', 'lower', or 'upper'.")
            
        removed_outliers_df = df.drop(index = outlier_indices_to_be_removed)
        display_text(f"Total Number of Outlier Entries Removed in {col}: {len(outlier_indices_to_be_removed)}", font_size = 18)
        print()
        return removed_outliers_df
    
    # Simply return the original DataFrame if remove = False
    return df


def train_and_evaluate_models(models_dictionary: dict[str, dict],
                              num_features: int,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: Union[pd.Series, np.ndarray],
                              y_test: Union[pd.Series, np.ndarray]) -> dict:
    """
    Trains and evaluates the passed in regression models using grid search for hyperparameter tuning.

    This function iterates through a dictionary of models, performs grid search cross-validation
    for hyperparameter tuning, trains each model with the best parameters, and evaluates their
    performance on both training and test sets. Information from this process is saved for each
    model in the model_results dictionary to be used later for model comparison.

    Args:
        models_dict (str, dict[str, dict]): A dictionary containing model information.
            Each key is a model name, and each value is another dictionary with 'model' and 'param_grid' keys.
            Defaults to the global 'models' variable.
        X_train (pd.DataFrame): The feature matrix for training.
            Defaults to the global 'X_train' variable.
        X_test (pd.DataFrame): The feature matrix for testing.
            Defaults to the global 'X_test' variable.
        y_train (Union[pd.Series, np.ndarray]): The target values for training.
            Defaults to the global 'y_train' variable.
        y_test (Union[pd.Series, np.ndarray]): The target values for testing.
            Defaults to the global 'y_test_preprocessed' variable.

    Returns:
        dict: A dictionary containing the results for each model. Each key is a model name,
        and each value is another dictionary with the following keys:
            * 'best_model': The trained model with the best hyperparameters.
            * 'best_params': The best hyperparameters found by grid search.
            * 'best_score': The best score achieved during grid search.
            * 'cv_results': The cross-validation results from grid search.
            * 'tune_train_time': The time taken for hyperparameter tuning and training.
            * 'y_train_predictions': Predictions on the training set.
            * 'y_test_predictions': Predictions on the test set.
            * 'train_mse': Mean Squared Error on the training set.
            * 'test_mse': Mean Squared Error on the test set.
            * 'test_mae': Mean Absolute Error on the test set.
            * 'test_r2': R-squared score on the test set.
            * 'test_adjusted_r2': Adjusted R-squared score on the test set.
            * 'intercept': The intercept of the model (for Linear, Lasso, and Ridge models only).
            * 'coefficients': The coefficients of the model (for Linear, Lasso, and Ridge models only).
    """
    model_results: dict = {}
    
    for model_name, model_data in models_dictionary.items():
        display_text(f"Training and evaluating: {model_name}", font_size = 16, font_weight = 'bold')
        
        model = model_data['model']
        param_grid = model_data['param_grid']
        
        # Users can configure the number of splits for cross-validation by changing the n_splits parameter
        cross_validation = KFold(n_splits = 5, shuffle = True, random_state = 42)
        grid_search = GridSearchCV(model, param_grid, cv = cross_validation, scoring = 'neg_mean_squared_error')
        tune_train_start_time = time.time()
        grid_search.fit(X_train, y_train)
        tune_train_end_time = time.time()
        tune_train_time = tune_train_end_time - tune_train_start_time
        display_markdown(md(f"* Hyperparameter Tuning and Model Training Time: {tune_train_time:.2f} seconds"))
        
        best_model = grid_search.best_estimator_
        
        y_train_predictions = best_model.predict(X_train)
        y_test_predictions = best_model.predict(X_test)
        
        test_r2 = r2_score(y_test, y_test_predictions)
        train_mse = mean_squared_error(y_train, y_train_predictions)
        test_mse = mean_squared_error(y_test, y_test_predictions)
        test_mae = mean_absolute_error(y_test, y_test_predictions)
        
        num_test_entries = len(y_test)
        test_adjusted_r2 = 1 - ((1 - test_r2) * (num_test_entries - 1) / (num_test_entries - num_features - 1))
        
        display_markdown(md(f"* Test R-squared: {test_r2:.{NUM_DECIMAL_PLACES}f}"))
        display_markdown(md(f"* Train MSE: {train_mse:.{NUM_DECIMAL_PLACES}f}"))
        display_markdown(md(f"* Test MSE: {test_mse:.{NUM_DECIMAL_PLACES}f}"))
        display_markdown(md(f"* Test MAE: {test_mae:.{NUM_DECIMAL_PLACES}f}"))
        display_markdown(md(f"* Test Adjusted R^2: {test_adjusted_r2:.{NUM_DECIMAL_PLACES}f}"))
        
        model_results[model_name] = {
            'best_model': best_model,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
            "tune_train_time": tune_train_time,
            "y_train_predictions": y_train_predictions,
            'y_test_predictions': y_test_predictions,
            'train_mse': train_mse,
            'test_mse': test_mse,
            "test_mae": test_mae,
            'test_r2': test_r2,
            "test_adjusted_r2": test_adjusted_r2
        }
        
        if model_name in ['Linear', 'Lasso', 'Ridge']:
            model_results[model_name]["intercept"] = best_model[-1].intercept_
            model_results[model_name]["coefficients"] = best_model[-1].coef_
    
    return model_results


def plot_comparative_model_performance(model_results: dict[str, dict]) -> None:
    """
    Create a comparative visualization of model performance metrics.

    This function generates a plot with two subplots comparing the performance of multiple regression models.
    The top subplot shows R-squared scores, while the bottom subplot displays Mean Squared Error (MSE)
    for both training and testing sets.

    Args:
        model_results (dict[str, dict]): A dictionary containing performance metrics for each model.
            The outer dictionary keys are model names, and the inner dictionary contains the following keys:
            * 'test_r2': R-squared score on the test set
            * 'train_mse': Mean Squared Error on the training set
            * 'test_mse': Mean Squared Error on the test set
            Defaults to the global 'model_results' variable.

    Returns:
        None
    """
    models = list(model_results.keys())

    # Create lists of performance metrics for plotting
    r2_scores = [model_results[model]['test_r2'] for model in models]
    train_mse_scores = [model_results[model]['train_mse'] for model in models]
    test_mse_scores = [model_results[model]['test_mse'] for model in models]

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
    fig.suptitle('Comparative Model Performance', fontsize=24)

    # Plot R-squared scores
    ax1.bar(models, r2_scores, color='skyblue')
    ax1.set_ylabel('R-squared Score')
    ax1.set_title('R-squared Scores by Model (Higher is Better)', fontweight='bold')
    ax1.set_xticks(np.arange(len(models)))  # Set fixed number of ticks
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    # Adjust the y-axis to start just below the minimum R-squared score
    r2_min = min(r2_scores) - 0.05
    r2_max = min(max(r2_scores) + 0.05, 1.00)
    ax1.set_ylim([r2_min, r2_max])

    # Add value labels on the R-squared bars
    for index, score in enumerate(r2_scores):
        ax1.text(index, score, f'{score:.5f}', ha='center', va='bottom')

    # Plot MSE and MAE scores
    x = np.arange(len(models))
    width = 0.35
    ax2.bar(x - (width / 2), train_mse_scores, width, label='Train MSE', color='salmon')
    ax2.bar(x + (width / 2), test_mse_scores, width, label='Test MSE', color='lightgreen')
    ax2.set_ylabel('Error Score')
    ax2.set_title('MSE During Training and Testing by Model (Lower is Better)', fontweight='bold')
    ax2.set_xticks(x)  # Set fixed number of ticks
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()

    # Add value labels on the MSE and MAE bars
    for index, (train_mse, test_mse) in enumerate(zip(train_mse_scores, test_mse_scores)):
        ax2.text(index - (width / 2), train_mse, f'{train_mse:.5f}', ha='center', va='bottom')
        ax2.text(index + (width / 2), test_mse, f'{test_mse:.5f}', ha='center', va='bottom')

    # Adjust layout and display the plot
    plt.tight_layout(rect = [0, 0, 1, 0.975])  # Add space at the top for the main title
    plt.subplots_adjust(hspace = 0.2)
    plt.show()
    
    
def summarize_results(results: dict) -> str:
    """
    This function creates a summary DataFrame of model performance metrics,
    displays it in a formatted table, and identifies the best performing model
    based on the R-squared score
    
    Args:
        results (dict): A dictionary containing performance metrics for each model.
            The keys are model names, and the values are dictionaries containing:
            * 'test_r2': R-squared score on the test set
            * 'train_mse': Mean Squared Error on the training set
            * 'test_mse': Mean Squared Error on the test set
            * 'test_mae': Mean Absolute Error on the test set
            * 'test_adjusted_r2': Adjusted R-squared score on the test set
            Defaults to the global 'model_results' variable.
    Returns:
        best_performing_model_name (str): The name of the best performing model based on the highest R-squared score.
    """ 
    summary_df = pd.DataFrame({
        'Model': results.keys(),
        'Test R-squared': [result['test_r2'] for result in results.values()],
        "Train MSE": [result['train_mse'] for result in results.values()],
        'Test MSE': [result['test_mse'] for result in results.values()],
        'Test MAE': [result['test_mae'] for result in results.values()],
        'Test Adjusted R-squared': [result['test_adjusted_r2'] for result in results.values()]
    })

    display_text("Model Performance Summary", font_size = 20, font_weight = "bold")
    
    display_df(summary_df)
    print()

    best_performing_model_name = summary_df.loc[summary_df['Test R-squared'].idxmax(), 'Model']
    display_markdown(md(f"### **Best Performing Model: {best_performing_model_name}** (based on R-squared score, the higher the better)"))
    display_markdown(md(f"* #### R-squared Score: {results[best_performing_model_name]['test_r2']}"))

    return best_performing_model_name


def plot_residuals_histograms_comparison(model_results: dict[str, dict],
                                         y_test: np.ndarray) -> None:
    """
    Plot histograms of residuals for multiple models vertically.
    
    Args:
        model_results (dict[str, dict]): Dictionary of model results, where each key is a model name
        and each value is a dictionary containing model predictions
        y_test (np.ndarray): Actual target values
    Returns:
        None
    """
    num_models = len(model_results)
    num_rows = math.ceil(num_models / 2)
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 7*num_rows))
    
    fig.suptitle('Comparison of Residuals Distributions', fontsize=24, y=1.02)
    
    axes_flat = axes.flatten() if num_models > 1 else [axes]
    
    # Determine global x-axis limits
    all_residuals = []
    for results in model_results.values():
        all_residuals.extend(y_test - results['y_test_predictions'])
    global_xlim = np.percentile(all_residuals, [1, 99])
    
    error_stats = {}
    for ax, (model_name, results) in zip(axes_flat, model_results.items()):
        y_pred = results['y_test_predictions']
        residuals = y_test - y_pred
        
        # Plot histogram and KDE
        sns.histplot(residuals, kde=True, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        
        ax.set_title(f'{model_name} Residuals', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Residuals (Actual - Predicted)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect Prediction')
        ax.set_xlim(global_xlim)
        
        # Calculate and display statistics
        mean_residual = np.mean(residuals)
        median_residual = np.median(residuals)
        std_residual = np.std(residuals)
        
        stats_text = (f'Mean: {mean_residual:.3f}\n'
                      f'Median: {median_residual:.3f}\n'
                      f'Std Dev: {std_residual:.3f}')
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=12)
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left')
        
        # Calculate and save statistics for percentage errors
        percentage_errors = (residuals / y_test) * 100
        mean_error = np.mean(percentage_errors)
        median_error = np.median(percentage_errors)
        std_error = np.std(percentage_errors)

        error_stats[model_name] = {
            'mean_error': mean_error,
            'median_error': median_error,
            'std_error': std_error
        }
    
    for ax in axes_flat[num_models:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.3, wspace=0.2)
    plt.show()
    return error_stats


def plot_residuals_scatter_comparison(model_results: dict[str, dict],
                                      y_test: np.ndarray) -> None:
    """
    This function generates a grid of scatter plots, one for each model in the input dictionary.
    Each plot visualizes the relationship between predicted values and their residuals,
    helping users better understand model performance and compare different models.

    Args:
        model_results (dict[str, dict]): A dictionary where each key is a model name and each value
            is another dictionary containing at least the key 'y_test_predictions', a numpy array
            of model predictions
        y_test (np.ndarray): Array of actual target values from the test set
    """
    num_models = len(model_results)
    num_rows = math.ceil(num_models / 2)
    fig, axes = plt.subplots(num_rows, 2, figsize=(22, 9*num_rows))  # Increased width for colorbars
    
    fig.suptitle('Comparison of Residuals Scatter Plots', fontsize=24, y=1.02)
    
    axes_flat = axes.flatten() if num_models > 1 else [axes]
    
    # Determine global axis limits and density range
    all_predicted = []
    all_residuals = []
    all_densities = []
    for results in model_results.values():
        y_pred = results['y_test_predictions']
        residuals = y_test - y_pred
        all_predicted.extend(y_pred)
        all_residuals.extend(residuals)
        xy = np.vstack([y_pred, residuals])
        all_densities.extend(stats.gaussian_kde(xy)(xy))
    
    global_xlim = np.percentile(all_predicted, [1, 99])
    global_ylim = np.percentile(all_residuals, [1, 99])
    vmin, vmax = np.percentile(all_densities, [5, 95])  # For consistent colorbar scale
    
    for ax, (model_name, results) in zip(axes_flat, model_results.items()):
        y_pred = results['y_test_predictions']
        residuals = y_test - y_pred
        
        # Create scatter plot with density coloring
        xy = np.vstack([y_pred, residuals])
        density = stats.gaussian_kde(xy)(xy)
        
        idx = density.argsort()
        x, y, z = y_pred[idx], residuals[idx], density[idx]
        
        scatter = ax.scatter(x, y, c=z, cmap='viridis', alpha=0.7, vmin=vmin, vmax=vmax)
        
        # Add trend line
        z = np.polyfit(y_pred, residuals, 1)
        p = np.poly1d(z)
        ax.plot(y_pred, p(y_pred), "black", alpha=0.8, label='Trend') 
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')  
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        
        ax.set_title(f'{model_name} Residuals', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Values', fontsize=14)
        ax.set_ylabel('Residuals', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right')
        
        # Add individual colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Density', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    
    for ax in axes_flat[num_models:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.97, hspace=0.2, wspace=0.1)
    
    plt.show()


def plot_actual_vs_predicted_comparison(model_results: dict[str, dict],
                                        y_test: np.ndarray,
                                        target_name: str = "Target") -> None:
    """
    Plots enhanced actual vs predicted values comparison for multiple regression models with improved range.
    
    Args:
    model_results (dict[str, dict]): Dictionary of model results, where each key is a model name
        and each value is a dictionary containing model predictions.
    y_test (np.ndarray): Actual target values.
    target_name (str): Name of the target variable for axis labels.
    
    Returns:
    None
    """
    num_models = len(model_results)
    num_rows = math.ceil(num_models / 2)
    fig, axes = plt.subplots(num_rows, 2, figsize=(20, 10*num_rows))
    fig.suptitle(f'Comparison of Actual vs Predicted {target_name} Values', fontsize=26, y=1.02)
    
    # Flatten axes array for easy iteration
    axes_flat = axes.flatten() if num_models > 2 else ([axes] if num_models == 1 else axes)
    
    # Calculate global min and max for consistent scaling
    all_y = np.concatenate([y_test] + [results['y_test_predictions'] for results in model_results.values()])
    global_min, global_max = np.min(all_y), np.max(all_y)
    
    # Calculate the range and add a buffer (e.g., 10% of the range on each side)
    data_range = global_max - global_min
    buffer = 0.1 * data_range
    plot_min = global_min - buffer
    plot_max = global_max + buffer
    
    # Ensure the aspect ratio is equal and adjust the limits if necessary
    center = (plot_min + plot_max) / 2
    half_range = max(plot_max - center, center - plot_min)
    plot_min = center - half_range
    plot_max = center + half_range
    
    # Calculate global min and max errors
    all_errors = np.concatenate([np.abs(y_test - results['y_test_predictions']) for results in model_results.values()])
    vmin, vmax = np.min(all_errors), np.max(all_errors)
    
    for ax, (model_name, results) in zip(axes_flat, model_results.items()):
        y_pred = results['y_test_predictions']
        errors = np.abs(y_test - y_pred)
        
        # Set background color to white and add grid
        ax.set_facecolor('white')
        ax.grid(True, linestyle=':', alpha=0.5, color='gray')
        
        # Create scatter plot with smaller, more transparent points
        scatter = ax.scatter(y_test, y_pred, c=errors, cmap='viridis', alpha=0.7, s=20, vmin=vmin, vmax=vmax)
        
        # Add diagonal line for perfect predictions
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Set consistent scale with balanced range
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        
        # Adjust title and labels
        ax.set_title(model_name, fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel(f'Predicted {target_name}', fontsize=14)
        ax.set_xlabel(f'Actual {target_name}', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add legend for perfect prediction line
        ax.legend(fontsize=10, loc='lower right')
        
        # Add colorbar for each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('Absolute Prediction Error Gradient', fontsize=12)
    
    # Hide any unused subplots
    for ax in axes_flat[num_models:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top = .97, hspace=0.2, wspace=0.25)
    plt.show()


def create_fit_function_equation_markdown(intercept: int,
                                          coefficient_list: list,
                                          feature_list: list,
                                          num_decimal_places: int = NUM_DECIMAL_PLACES) -> str:
    model_equation = f"### {num_decimal_places} = {intercept:.{num_decimal_places}f}"
    
    for i in range(len(coefficient_list)):
        model_equation += f" + {coefficient_list[i]:.{num_decimal_places}f}({feature_list[i]})"
        
    return model_equation


def display_model_evaluation_results(model_name: str, results: dict) -> None:
    """
    Displays the evaluation results for a given model.
    
    Args:
        model_name (str): Name of the model
        results (dict): Dictionary containing the evalutation results of each model
    Returns: None
    """
    display_markdown(md(f"### **Model: {model_name}**"))
    display_markdown(md(f"* #### **Train MSE:** {results[model_name]['train_mse']:.{NUM_DECIMAL_PLACES}f}"))
    display_markdown(md(f"* #### **Test MSE:** {results[model_name]['test_mse']:.{NUM_DECIMAL_PLACES}f}"))
    display_markdown(md(f"* #### **Test MAE:** {results[model_name]['test_mae']:.{NUM_DECIMAL_PLACES}f}"))
    display_markdown(md(f"* #### **Test R^2:** {results[model_name]['test_r2']:.{NUM_DECIMAL_PLACES}f}"))
    display_markdown(md(f"* #### **Test Adjusted R^2:** {results[model_name]['test_adjusted_r2']:.{NUM_DECIMAL_PLACES}f}"))
    
    if model_name in ['Linear', 'Lasso', 'Ridge']:
        display_markdown(md(f"### **Model Equation:**"))
        display_markdown(md(create_fit_function_equation_markdown(results[model_name]["intercept"], results[model_name]["coefficients"])))
        
    
def plot_residuals_histogram(model_name: str, residuals: np.ndarray) -> None:
    """
    Display a histogram of residuals with KDE and statistics.
    
    Args:
        residuals (np.ndarray): Residuals (actual - predicted)
        model_name (str): Name of the model
    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(residuals, kde=True, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Density')
    plt.title(f'Histogram of Residuals ({model_name})')
    plt.grid(True)
    
    mean_residual = np.mean(residuals)
    median_residual = np.median(residuals)
    std_residual = np.std(residuals)

    stats_text = (f'Mean: {mean_residual:.3f}\n'
                  f'Median: {median_residual:.3f}\n'
                  f'Std Dev: {std_residual:.3f}\n')
    
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12)
    
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.legend()
    plt.show()


def plot_residuals_scatter(model_name: str,
                           y_pred: np.ndarray,
                           residuals: np.ndarray) -> None:
    """
    Displays a scatter plot of residuals vs predicted values with density coloring.
    
    Args:
        y_pred (np.ndarray): Predicted target values
        residuals (np.ndarray): Residuals (actual - predicted)
        model_name (str): Name of the model
    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate density for coloring
    xy = np.vstack([y_pred, residuals])
    density = stats.gaussian_kde(xy)(xy)
    
    idx = density.argsort()
    x, y, z = y_pred[idx], residuals[idx], density[idx]
    
    scatter = plt.scatter(x, y, c=z, cmap='viridis', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(y_pred, residuals, 1)
    p = np.poly1d(z)
    plt.plot(y_pred, p(y_pred), "black", alpha=0.8, label='Trend')

    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Predicted Target Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'Residual Plot ({model_name})')
    plt.grid(True)
    plt.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Density', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    plt.show()


def plot_actual_vs_predicted(model_name: str,
                             y_test: np.ndarray,
                             y_pred: np.ndarray,
                             target_name: str = "Target Values") -> None:
    """
    Displays a plot of actual vs predicted values with error gradient.
    
    Args:
        y_test (np.ndarray): Actual target values
        y_pred (np.ndarray): Predicted target values
        model_name (str): Name of the model
        target_name (str): Name of the target variable for axis labels
    Returns:
        None
    """
    errors = np.abs(y_test - y_pred)
    plt.figure(figsize=(13, 6))
    scatter = plt.scatter(y_test, y_pred, c=errors, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Absolute Error')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel(f'Actual {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.title(f'Actual vs. Predicted with Error Gradient ({model_name})')
    plt.grid(True)

    # Move legend to the bottom right
    plt.legend(loc='lower right')
    
    plt.show()
    
    
def display_model_evaluation_and_plots(model_name: str,
                                       model_results: dict[str, dict],
                                       y_test: np.ndarray) -> None:
    """
    Display the evaluation results and plots for a given model by calling previous defined functions.
    
    Args:
        model_name (str): Name of the model
    Returns: None
    """
    y_test_predictions = model_results[model_name]["y_test_predictions"]
    residuals = y_test - y_test_predictions

    display_model_evaluation_results(model_name)
    plot_residuals_histogram(model_name, residuals)
    plot_residuals_scatter(model_name, y_test_predictions, residuals)
    plot_actual_vs_predicted(model_name, y_test, y_test_predictions)
    
    
def save_model_pickle(file_name: str,
                      model_name: str,
                      results: dict[str, dict]) -> None:
    """
    Saves the best performing model to a pickle file.
    
    Args:
        file_name (str): Name of the pickle file to save the model to
        model_name (str): Name of the model that is about to be saved
        results (dict[str, dict]): Dictionary containing the model object, predictions on the testing data, and other model perfomance data
    Returns:
        None
    """
    model_to_save = results[model_name]['best_model']
    with open(file_name, 'wb') as file:
        pickle.dump(model_to_save, file)
        

def save_model_joblib(file_name: str,
                      model_name: str,
                      results: dict[str, dict]) -> None:
    """
    Saves the best performing model to a joblib file.
    
    Args:
        file_name (str): Name of the joblib file to save the model to
        model_name (str): Name of the model that is about to be saved
        results (dict[str, dict]): Dictionary containing the model object, predictions on the testing data, and other model perfomance data
    Returns:
        None
    """
    model_to_save = results[model_name]['best_model']
    dump(model_to_save, file_name)
    

def save_model(file_name: str,
               model_name: str,
               results: dict[str, dict],
               method: str) -> None:
    """
    Saves the best performing model to a file using the specified method.
    
    Args:
        file_name (str): Name of the file to save the model to
        model_name (str): Name of the model to save
        results (dict[str, dict]): Dictionary containing the model object, predictions on the testing data, and other model perfomance data
        method (str): Method to use for saving the model ("pickle" or "joblib")
    Returns:
        None
    """
    if method == "pickle":
        save_model_pickle(file_name, model_name, results)
    elif method == "joblib":
        save_model_joblib(file_name, model_name, results)
    else:
        raise ValueError("Invalid method specified. Please use 'pickle' or 'joblib'.")
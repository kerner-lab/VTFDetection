import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, roc_curve, roc_auc_score, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from numpy import argmax, sqrt

def load_data(file_path):
    # Load the data from a CSV file
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    # Preprocess the data by dropping missing values and converting columns to appropriate data types
    df = df.dropna()
    df['Binary'] = df['Binary'].astype(int)
    df['Binary'] = df['Binary'].replace(2, 1)
    return df


def sample_data(df, sample_fraction=0.2):
    # Sample a fraction of data from each volcano group
    grouped = df.groupby('Volcano')
    sampled_dfs = []

    for _, group in grouped:
        sampled_df = group.sample(frac=sample_fraction)
        sampled_dfs.append(sampled_df)

    sampled_df = pd.concat(sampled_dfs)
    sampled_df = sampled_df.dropna()
    return sampled_df


def evaluate_unsampled_data(df, sampled_df):
    # Evaluate the unsampled data by finding the data that was not sampled
    unsampled = pd.merge(df, sampled_df, how="outer", indicator=True)
    unsampled = unsampled.loc[unsampled["_merge"] == "left_only"]
    unsampled.drop("_merge", axis=1, inplace=True)
    unsampled = unsampled.dropna()
    return unsampled


def calculate_roc_auc(true_thresh, pred_thresh):
    # Calculate ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(true_thresh, pred_thresh)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def calculate_f1_scores(true_thresh, pred_thresh, thresholds):
    # Calculate F1 score for each threshold and choose the best threshold
    f1_scores = []
    for threshold in thresholds:
        y_pred = np.where(pred_thresh > threshold, 1, 0)
        f1_scores.append(f1_score(true_thresh, y_pred))

    best_f1_score = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_f1_score, best_threshold


def calculate_gmeans_threshold(true_thresh, pred_thresh, thresholds, tpr, fpr):
    # Calculate G-Mean for each threshold and choose the best threshold
    gmeans = sqrt(tpr * (1 - fpr))
    ix = argmax(gmeans)
    gmeans_threshold = thresholds[ix]
    print("Best Threshold=%f, G-Mean=%.3f" % (thresholds[ix], gmeans[ix]))
    return gmeans_threshold


def calculate_metrics(true_eval, pred_eval, f1_threshold, gmeans_threshold):
    # Calculate metrics such as accuracy and confusion matrix
    threshold = (f1_threshold * 0.5 + gmeans_threshold * 0.5)
    y_pred_class = np.array(pred_eval) > threshold
    cm = confusion_matrix(true_eval, y_pred_class)
    accuracy = accuracy_score(true_eval, y_pred_class)
    return accuracy, cm, threshold


def save_results(df, threshold, output_path):
    # Save the results to a CSV file
    df['Decision'] = df['Prediction'].apply(lambda x: 1.0 if x >= threshold else 0.0)
    df.to_csv(output_path, index=False)


def main():
    # Path to the input CSV file 
    file_path = "/Users/adityamohan/Documents/RA/csv/merged_df_pred_proc.csv"
    # Output file path for final results
    output_path = "final_pred_proc.csv"

    # Load and preprocess the data
    df = load_data(file_path)
    df = preprocess_data(df)

    # Sample a fraction of data
    sampled_df = sample_data(df)

    # Evaluate unsampled data
    unsampled_df = evaluate_unsampled_data(df, sampled_df)

    # Calculate ROC AUC score
    roc_auc = calculate_roc_auc(sampled_df['Binary'], sampled_df['Prediction'])
    print("ROC AUC Score:", roc_auc_score(sampled_df['Binary'], sampled_df['Prediction']))

    fpr, tpr, thresholds = roc_curve(sampled_df['Binary'], sampled_df['Prediction'])
    # Calculate F1 scores
    f1_score, f1_threshold = calculate_f1_scores(unsampled_df['Binary'], unsampled_df['Prediction'],thresholds)
    print("Best F1 Score:", f1_score)
    print("Best F1 Threshold:", f1_threshold)

    # Calculate G-Mean
    gmeans_threshold = calculate_gmeans_threshold(sampled_df['Binary'], sampled_df['Prediction'], thresholds, tpr, fpr)
    print("Best G-Mean Threshold:", gmeans_threshold)

    # Calculate metrics
    accuracy, confusion_matrix, threshold = calculate_metrics(unsampled_df['Binary'], unsampled_df['Prediction'], f1_threshold, gmeans_threshold)
    print("Accuracy: ", accuracy)
    print("Confusion Matrix:\n", confusion_matrix)

    print(" Chosen threshold: ",threshold)

    # Save the results to a CSV file
    save_results(df, threshold, output_path)


if __name__ == "__main__":
    main()

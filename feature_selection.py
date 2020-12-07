import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC, SVR
import glob, os
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import codecs, json

# Read all the json files and combine all of them into a single dataframe
def readData(path):
    json_path = path
    df = pd.concat(map(pd.read_json, glob.glob(os.path.join('',json_path))))
    X, y = df.iloc[:,1:], df.iloc[:,0]
    return X,y

# Calculate the correlation coefficient between features and keep the one between highly correctled features
def corr_features(X):
    correlated_features = set()
    correlation_matrix = X.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    # index_select = [x - 1 for x in list(correlated_features)]
    # print(correlated_features)
    newX = X.drop(correlated_features,axis=1)
    # print(newX)
    return newX

# It will plot the optimal number of feature based on the randon forest model
def optimal_num_features(X,y):
    print("Ploting optimal number of features...")
    # rf = RandomForestClassifier()
    rf = RandomForestClassifier(min_samples_split=10, min_samples_leaf=10,random_state=101)
    rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(5), scoring='accuracy')
    rfecv.fit(X, y)
    # rfecv.show()
    plt.figure(figsize=(16, 9))
    plt.title('Recursive Feature Elimination with Cross-Validation (REFCV)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('Score', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
    plt.show()
    # print(rfecv.estimator_.feature_importances_)
    return rfecv

# It will polt the features importance graph 
def plotFeaturesRanking(rfecv,X):
    print("Plotting features ranking...")
    features = X.columns
    importances = rfecv.estimator_.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(25,8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

def main():
    json_path = './processed_data/all/*'
    X,y = readData(json_path)
    newX = corr_features(X)

    # plotting
    rfecv = optimal_num_features(newX,y)
    plotFeaturesRanking(rfecv,newX)
    # Saving
    newX = newX.drop(columns=[14,23])
    results = pd.concat([y, newX], axis=1)
    results = results.values.tolist()
    json.dump(results, codecs.open('all.json', 'w', encoding='utf-8'), \
        separators=(',', ':'), sort_keys=True, indent=2)
    

if __name__ == '__main__':
    main()
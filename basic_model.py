import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVC, SVR
import glob, os
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Plot the value distribution 
def plotValueDist(y):
    y_count = y.groupby(['0']).size().tolist()
    labels = 'Confused', 'Uncertain', 'Not Confused'
    explode = (0, 0.1, 0)  # only "explode" the 2nd slice

    fig1, ax1 = plt.subplots()
    ax1.pie(y_count, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.show()

# Find the best parameters for svm
def svmGridSearch(X,y):
    steps = [('scaler', StandardScaler()), ('SVM', SVC())]
    hyperF = {'SVM__C':np.logspace(-3, 2, 6), 'SVM__gamma':np.logspace(-3, 2, 6)}
    pipeline = Pipeline(steps) # define the pipeline object.
    clf = GridSearchCV(pipeline, param_grid=hyperF, cv = 5, verbose = 1, 
                         n_jobs = -1)
    bestF = clf.fit(X, y)
    print(bestF.score(X, y))
    print(bestF.best_params_)
    
# Find the best parameter for knn
def knnGridSearch(X_train,y_train,X_valid,y_valid):
    n_neighbors = list(range(1,20))
    #Convert to dictionary
    hyperparameters = dict(n_neighbors=n_neighbors)
    #Create new KNN object
    knn_2 = KNeighborsClassifier()
    #Use GridSearch
    clf = GridSearchCV(knn_2, hyperparameters, cv=2)
    #Fit the model
    best_model = clf.fit(X_train,y_train)
    print(best_model.score(X_valid,y_valid))
    #Print The value of best Hyperparameters
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

def five_cv(model,X,y,name):
    scores = cross_val_score(model, X, y, cv=5)
    print(name + " 5-CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def main():
    csv_path = './processed_data/csv/all/*'
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('',csv_path))))
    X, y = df.iloc[:,2:], df.iloc[:,1]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    print("Date was loaded...")
    # print(y)
    # SVM 
    svm_all = make_pipeline(
    MinMaxScaler(),
    SVC(kernel='linear',C=10.0,gamma=0.01)
    )
    svm_all.fit(X_train, y_train)
    print("SVM accuracy: ", svm_all.score(X_train, y_train))
    print("SVM val accuracy: ", svm_all.score(X_valid, y_valid))
    five_cv(svm_all,X,y, 'SVM')
    # KNN
    # knnGridSearch(X_train,y_train,X_valid,y_valid)
    knn_model = make_pipeline(
        MinMaxScaler(),
        KNeighborsClassifier(n_neighbors=2)
    )
    knn_model.fit(X_train, y_train)
    print("Knn accuracy: ", knn_model.score(X_train, y_train))
    print("Knn val accuracy: ", knn_model.score(X_valid, y_valid))
    five_cv(knn_model,X,y, 'KNN')

    rf = RandomForestClassifier(n_estimators=200,
        max_depth=9, min_samples_split=10, min_samples_leaf=10)
    rf.fit(X_train, y_train)
    print("Random Forest accuracy: ", rf.score(X_train, y_train))
    print("Random Forest validation accuracy: ", rf.score(X_valid, y_valid))
    five_cv(rf,X,y, 'Random Forest')

    # # Plotting feature importance
    # # features = X.columns
    # # importances = rf.feature_importances_
    # # indices = np.argsort(importances)
    # # plt.figure(figsize=(25,8))
    # # plt.title('Feature Importances')
    # # plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    # # plt.yticks(range(len(indices)), [features[i] for i in indices])
    # # plt.xlabel('Relative Importance')
    # # plt.show()

   

    plotValueDist(df)
    

if __name__ == '__main__':
    main()

#Mushroom Classification

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
dataset= pd.read_csv('mushrooms.csv')
X= dataset.iloc[:, 1:]
y= dataset.iloc[:, 0].values

#Encoding variables
from sklearn.preprocessing import LabelEncoder
labelencoder_y= LabelEncoder()
y= labelencoder_y.fit_transform(y)

X= pd.get_dummies(X, drop_first= True)
X= X.values

#Splitting into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2)

#Dimensionalit Reduction
from sklearn.decomposition import PCA
pca= PCA(n_components= 2)
X_train= pca.fit_transform(X_train)
X_test= pca.transform(X_test)

#Fitting SVM to training set
from sklearn.svm import SVC
classifier= SVC(C= 100, kernel= 'rbf', gamma= 5.0)
classifier.fit(X_train, y_train)

#Predicting values for test set
y_pred= classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)

#Calculating accuracy
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator= classifier, X= X_train, y= y_train, cv= 10)
mean= accuracies.mean()
std= accuracies.std()

#Finding best parameters
from sklearn.model_selection import GridSearchCV
parameters= [{'C': [1, 10, 100], 'kernel': ['linear']},
             {'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.1, 0.5, 1.0, 5.0]}]
grid_search= GridSearchCV(estimator= classifier, param_grid= parameters, scoring= 'accuracy',
                          cv= 10, n_jobs= -1)
grid_search= grid_search.fit(X_train, y_train)
best_parameters= grid_search.best_params_

#Visualising training set results
from matplotlib.colors import ListedColormap
X_set, y_set= X_train, y_train
X1, X2= np.meshgrid(np.arange(start= X_set[:, 0].min() - 1, stop= X_set[:, 0].max() + 1, step= 0.01),
                    np.arange(start= X_set[:, 1].min() - 1, stop= X_set[:, 1].max() + 1, step= 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha= 0.75, cmap= ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], c= ListedColormap(('red', 'green'))(i),
                label= j)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Mushroom Classification (Test Set)")
plt.legend()
plt.show()

#Visualising test set results
from matplotlib.colors import ListedColormap
X_set, y_set= X_test, y_test
X1, X2= np.meshgrid(np.arange(start= X_set[:, 0].min() - 1, stop= X_set[:, 0].max() + 1, step= 0.01),
                    np.arange(start= X_set[:, 1].min() - 1, stop= X_set[:, 1].max() + 1, step= 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha= 0.75, cmap= ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], c= ListedColormap(('red', 'green'))(i),
                label= j)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Mushroom Classification (Test Set)")
plt.legend()
plt.show()
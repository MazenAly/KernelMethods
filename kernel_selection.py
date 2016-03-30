from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_classification

import matplotlib.pyplot as plt

#creating a dataset of 1000 examples
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=2
                           , n_clusters_per_class=2,random_state=0)
X = DataFrame(X, columns={'dim1' , 'dim2'})
y = DataFrame(y, columns={'target'})

def get_Z(clf , xx , yy , X ,y):
    clf.fit(X, y['target'])
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return Z

h=.02

# create a mesh to plot in
x_min, x_max = X.dim1.min()-1, X.dim1.max()+1
y_min, y_max = X.dim2.min()-1, X.dim2.max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


kernels_scores = []
kernels = ['linear' , 'poly' , 'rbf' ]
plt.set_cmap(plt.cm.Paired)
clf = SVC( kernel='linear')
scores = cross_val_score(clf, X, y['target'], cv=10 , scoring='roc_auc')
kernels_scores.append( scores.mean() )
Z = get_Z(clf , xx , yy , X ,y)
#pl.set_cmap(pl.cm.Paired)
plt.contourf(xx, yy, Z , cmap=plt.cm.Paired)
plt.axis('tight')

# Plot also the training points
plt.scatter(X.dim2,  X.dim1 , c=y  , cmap=plt.cm.Paired)

plt.title("Linear kernel with 10 CV AUC of " + str(round(scores.mean() , 3) ) )
plt.show()  


clf = SVC( kernel='poly' )
scores = cross_val_score(clf, X, y['target'], cv=10 , scoring='roc_auc')
kernels_scores.append( scores.mean() )

Z = get_Z(clf , xx , yy , X ,y)
#pl.set_cmap(pl.cm.Paired)
plt.contourf(xx, yy, Z)
plt.axis('tight')

# Plot also the training points
plt.scatter(X.dim2,  X.dim1 , c=y  , cmap=plt.cm.Paired)

plt.title("Poly kernel with 10 CV AUC of " + str(round(scores.mean() , 3) ) )
plt.show()  

clf = SVC( kernel='rbf' )
scores = cross_val_score(clf, X, y['target'], cv=10 , scoring='roc_auc')
kernels_scores.append( scores.mean() )


Z = get_Z(clf , xx , yy , X ,y)
plt.set_cmap(plt.cm.Paired)
plt.contourf(xx, yy, Z)
plt.axis('tight')

# Plot also the training points
plt.scatter(X.dim2, X.dim1, c=y)

plt.title("RBF kernel with 10 CV AUC of " + str(round(scores.mean() , 3) ) )
plt.show()  


plt.bar(np.arange(len(kernels)), kernels_scores , align='center' )
plt.title('Kernel types and their AUC')
plt.xticks(np.arange(len(kernels)), kernels)
plt.xlabel('Kernels')
plt.ylabel('AUC')
plt.ylim(0.95,1)
plt.yticks(np.linspace(0.95,1,10,endpoint=True))
plt.show()


gamma_C_values = [ -8 , -2 , 2 , 6, 15 ]
gamma_scores = []
C_scores = [] 

j=1
plt.set_cmap(plt.cm.Paired)
for i in gamma_C_values:
    plt.set_cmap(plt.cm.Paired)
    clf = SVC( kernel='rbf' , gamma =   2**i)
    scores = cross_val_score(clf, X, y['target'], cv=10 , scoring='roc_auc')
    gamma_scores.append( scores.mean() ) 
    plt.subplot(2, 5,j)
    Z = get_Z(clf , xx , yy , X ,y)
    plt.set_cmap(plt.cm.Paired)
    plt.contourf(xx, yy, Z)
    plt.axis('tight')
    plt.scatter(X.dim2, X.dim1, c=y)
    plt.title("gamma=2^" + str(i) +  " AUC :" + str(round(scores.mean() , 3) ) )
    
    plt.set_cmap(plt.cm.Paired)
    
    clf2 = SVC( kernel='rbf' , C=  2**i)
    scores2 = cross_val_score(clf2, X, y['target'], cv=10 , scoring='roc_auc')
    C_scores.append( scores2.mean() )
    
    plt.subplot(2, 5,j+5)
    Z = get_Z(clf2 , xx , yy , X ,y)
    plt.set_cmap(plt.cm.Paired)
    plt.contourf(xx, yy, Z)
    plt.axis('tight')
    plt.scatter(X.dim2, X.dim1, c=y)
    plt.title("C=2^" + str(i) +  " AUC :" + str(round(scores2.mean() , 3) ) )  
    j += 1 

plt.show() 

#Plotting the bar-chart for the gamma and C values and their respective AUC score
print gamma_scores
print C_scores

plt.figure(1)
plt.subplot(211)
plt.bar(np.arange(len(gamma_C_values)), gamma_scores , align='center' )
plt.title('Gamma and C parameters and their AUC')
plt.xlabel('gamma values')
plt.ylabel('AUC')
plt.xticks(np.arange(len(gamma_C_values)), ['2^' + str(x) for x in gamma_C_values ])
plt.yticks(np.linspace(0.5,1,10,endpoint=True))
plt.ylim(0.5,1)

plt.subplot(212)
plt.bar(np.arange(len(gamma_C_values)), C_scores , align='center' )
plt.xlabel('C values')
plt.ylabel('AUC')
plt.xticks(np.arange(len(gamma_C_values)), ['2^' + str(x) for x in gamma_C_values ])
plt.yticks(np.linspace(0.9,1,10,endpoint=True))
plt.ylim(0.9,1)
plt.show()
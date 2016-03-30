from openml.apiconnector import APIConnector
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
apikey = 'b6da739f426042fa9785167b29887d1a'
connector = APIConnector(apikey=apikey)
dataset = connector.download_dataset(59)



columns_names = [  'a' + str(x) for x in range(1,35) ]
columns_names.append('target')
train = dataset.get_dataset()
train = pd.DataFrame(train , columns = columns_names ) 
y = train['target']
X = train.iloc[:,:-1]

print X 
print y.value_counts()



gamma_range = np.logspace(-15, 15, 20, base=2)
param_grid =   dict(gamma=gamma_range,  kernel= ['rbf'] )
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=10 , scoring='roc_auc')
grid.fit(X, y)
scores1 = [x[1] for x in grid.grid_scores_]
print gamma_range
print scores1


plt.bar(np.arange(len(gamma_range)), scores1 , align='center' )
plt.xlim(-1, len(gamma_range))
plt.xticks(np.arange(len(gamma_range)), [ '2^' + str(round(np.log2(x), 1)) for x in gamma_range ] , rotation=45 )
plt.title('Gamma values and their AUC')
plt.xlabel('gamma values')
plt.ylabel('AUC')
plt.show()


gamma_range = np.logspace(-15, 15, 30, base=2)
C_range = np.logspace(-15, 15, 30, base=2)

param_grid =dict(C= C_range , gamma=gamma_range,  kernel= ['rbf'] )
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=10 , scoring='roc_auc')
grid.fit(X, y)
scores = [x[1] for x in grid.grid_scores_]
scores


scores = [x[1] for x in grid.grid_scores_]
scores = np.array(scores).reshape(len(C_range), len(gamma_range))
scores


plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), [ '2^' + str(round(np.log2(x), 1)) for x in gamma_range ], rotation=45)
plt.yticks(np.arange(len(C_range)), [ '2^' + str(round(np.log2(x), 1)) for x in C_range ])
plt.title('Validation accuracy')
plt.show()


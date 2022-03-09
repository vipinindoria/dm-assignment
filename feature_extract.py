import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

np.random.seed(100)

df = pd.read_csv("data/raw/kidneyChronic.csv")
df = df.replace({'?': np.nan, '\t?': np.nan})

class LabelEncoderByCol(BaseEstimator, TransformerMixin):
    # Initialize a LabelEncoder for each column separately
    def __init__(self,col):
        self.col = col
        self.le_dic = {}
        for el in self.col:
            self.le_dic[el] = LabelEncoder()

    def fit(self,x,y=None):
        # Only using non-NaN values in the encoder
        x[self.col] = x[self.col].fillna('NaN')
        for el in self.col:
            a = x[el][x[el]!='NaN']
            self.le_dic[el].fit(a)
        return self

    def transform(self,x,y=None):
        # Only transforming the non-NaN values
        x[self.col] = x[self.col].fillna('NaN')
        for el in self.col:
            a = x[el][x[el]!='NaN']
            b = x[el].to_numpy()
            b[b!='NaN'] = self.le_dic[el].transform(a)
            x[el]=b
        return x

# Running the Label Encoder on the Categorical Variables
catvar = ['sg','al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','ane','pe','class']
le = LabelEncoderByCol(col=catvar)
le = le.fit(df)
df_i = le.transform(df)

print(df_i)

# Using KNN to impute missing data in the dataset
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
dft = imputer.fit_transform(df_i)
df = pd.DataFrame(data=dft, columns=df_i.columns)
print(df)

# Eliminating features with high correlation
corr = df.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False

selected_columns = df.columns[columns]
df = df[selected_columns]
print(df)

# Checking to see how many features to select by using ANOVA F-values
# k=5 Features were deemed the best based on inflection point of train vs test error on SVC
train_accuracy_list = []
test_accuracy_list = []
for K in range(1,15):
    fs = SelectKBest(score_func=f_classif, k=K)
    selected = fs.fit_transform(df.iloc[:,:24], df.iloc[:,24])

    select_flip = list(map(list, zip(*selected)))
    best_col = []
    for i in select_flip:
        best_col.append(df.columns[(df.values == np.asarray(i)[:,None]).all(0)][0])

    data = df[best_col]
    result = df['class']

    x_train, x_test, y_train, y_test = train_test_split(data.values, result.values, test_size = 0.2)
    svc = SVC()
    svc = svc.fit(x_train, y_train)
    train_prediction = svc.predict(x_train)
    cm = confusion_matrix(y_train, train_prediction)
    sum = 0
    for i in range(cm.shape[0]):
        sum += cm[i][i]

    accuracy = sum/x_train.shape[0]
    train_accuracy_list.append(accuracy)
    test_prediction = svc.predict(x_test)
    cm = confusion_matrix(y_test, test_prediction)
    sum = 0
    for i in range(cm.shape[0]):
        sum += cm[i][i]

    accuracy = sum/x_test.shape[0]
    test_accuracy_list.append(accuracy)

X = range(1, 15)
plt.plot(X, test_accuracy_list)
plt.plot(X, train_accuracy_list)
plt.show()

# selected_columns = selected_columns[:24].values
#
# import statsmodels.api as sm
# def backwardElimination(x, Y, sl, columns):
#     numVars = len(x[0])
#     for i in range(0, numVars):
#         regressor_OLS = sm.OLS(Y, x).fit()
#         maxVar = max(regressor_OLS.pvalues).astype(float)
#         if maxVar > sl:
#             for j in range(0, numVars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                     x = np.delete(x, j, 1)
#                     columns = np.delete(columns, j)
#
#     return x, columns
#
#
# SL = 0.05
# data_modeled, selected_columns = backwardElimination(df.iloc[:,:24].values, df.iloc[:,24].values, SL, selected_columns)
# result = pd.DataFrame()
# result['diagnosis'] = df.iloc[:,24]
# data = pd.DataFrame(data = data_modeled, columns = selected_columns)
#
# fig = plt.figure(figsize = (20, 25))
# j = 0
# for i in data.columns:
#     print(i)
#     plt.subplot(6, 5, j+1)
#     j += 1
#     sns.distplot(data[i][result['diagnosis']==0], color='g', label = 'ckd')
#     sns.distplot(data[i][result['diagnosis']==1], color='r', label = 'notckd')
#     plt.legend(loc='best')
# fig.tight_layout()
# fig.subplots_adjust(top=0.95)
# plt.show()

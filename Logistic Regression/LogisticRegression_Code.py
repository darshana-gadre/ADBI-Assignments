import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.feature_selection import chi2

import warnings
warnings.filterwarnings("ignore")


df = pd.read_excel('eBayAuctions.xls')
df.head()


#Train Test Split
train_set, val_set = train_test_split(df, train_size=0.6, test_size=0.4, shuffle=False)


#Pivot Table for average of 'Competitive' as a function of predictor 'Category'
pd.pivot_table(train_set,index=["Category"], values=["Competitive?"], aggfunc=np.mean)

#Pivot Table for average of 'Competitive' as a function of predictor 'currency'
pd.pivot_table(train_set,index=["currency"], values=["Competitive?"], aggfunc=np.mean)

##Pivot Table for average of 'Competitive' as a function of predictor 'andDay'
pd.pivot_table(train_set,index=["endDay"], values=["Competitive?"], aggfunc=np.mean)


#Dictionary with combined categoeries. Categories with difference in mean less than or equal to 0.5 are combined into one category.
combinedDict = {"Category" : {"Antique/Art/Craft":"Category1", "Business/Industrial":"Category1", "Collectibles":"Category1", "Computer":"Category1", 
                "Automotive":"Category2", "Books":"Category2", "Coins/Stamps":"Category2", "Jewelry":"Category2", "Toys/Hobbies":"Category2", 
                 "Music/Movie/Game":"Category3", 
                "Pottery/Glass":"Category4", 
                 "Clothing/Accessories":"Category5", 
                 "Electronics":"Category6", "Home/Garden":"Category6", "SportingGoods":"Category6",
                 "EverythingElse":"Category7","Health/Beauty":"Category7",
                 "Photography":"Category8"
                }, 
                
                "currency" : {"EUR":"Currency1", "US":"Currency1", 
                "GBP":"Currency2"}, 
                
                "endDay" : {"Mon":"Day1", "Thu":"Day1", "Tue":"Day1", 
                 "Sat":"Day2", 
                 "Sun":"Day3", 
                 "Fri":"Day4", "Wed":"Day4"}
               }



#Replace training data with combined categories
train_set.replace(combinedDict, inplace=True)
train_set.head()


#Creating dummy variables for categorical predictors in train data
dummies_train_set = pd.get_dummies(train_set)
dummies_train_set.head()


#Replace testing data with combined categories
val_set.replace(combinedDict, inplace=True)
val_set.head()


#Creating dummy variables for categorical predictors in test data
dummies_val_set = pd.get_dummies(val_set)
dummies_val_set.head()


#Logistic Regression Model for all predictors

#Separate predictor and response variables of training data
x_train = dummies_train_set.loc[:, dummies_train_set.columns != 'Competitive?']
y_train = dummies_train_set['Competitive?']

#Building regression model with all predictors
fit_all = LogisticRegression().fit(x_train, y_train)


#Separate predictor and response variables of testing data
x_test = dummies_val_set.loc[:, dummies_val_set.columns != 'Competitive?']
y_test = dummies_val_set['Competitive?']


#Getting predictions on testing data
y_pred=fit_all.predict(x_test)


#Evaluating model
score = fit_all.score(x_test, y_test)
print('\nAccuracy of model with all predictors : ',score)


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix of model with all predictors : \n',cnf_matrix)


print('\nIntercept of model with all predictors : ',fit_all.intercept_)


#Get all regression coefficients
all_coef = list(fit_all.coef_[0])

#Get significance values of predictors
stats, p_values = chi2(x_train, y_train)

significance_values = []

for i in range(0, 18):
    significance_values.append([round(abs(all_coef[i]), 4), x_train.columns[i], round(p_values[i], 4)])
    
significance_values.sort()

#print(significance_values)


significant_predictors = []

#Choose predictors with significance values above 0.05
for val in significance_values:
    if val[2] <= 0.05:
        significant_predictors.append(val[1])

print('\nSignificant Predictors : ',significant_predictors)


#Logistic Regression Model for Single Predictor

x_train_single = dummies_train_set[['OpenPrice']]

y_train_single = dummies_train_set['Competitive?']

#Building regression model with single predictors
fit_single = LogisticRegression().fit(x_train_single, y_train_single)


print('\nCoefficient of model with Single Predictor : ',fit_single.coef_)
print('\nIntercept of model with Single Predictor : : ',fit_single.intercept_)


x_test_single = dummies_val_set[['OpenPrice']]

y_test_single = dummies_val_set['Competitive?']

y_pred_single=fit_single.predict(x_test_single)


score_single = fit_single.score(x_test_single, y_test_single)
print('\nAccuracy of model with Single Predictor : ',score_single)


cnf_matrix1 = metrics.confusion_matrix(y_test_single, y_pred_single)
print('\nConfusion Matrix of model with Single Predictor : \n',cnf_matrix1)


#Reduced Logistic Regression Model. Model with predictors having significance values greater than 0.05
x_train_reduced = dummies_train_set[significant_predictors]

y_train_reduced = dummies_train_set['Competitive?']

#Building reduced regression model
fit_reduced = LogisticRegression().fit(x_train_reduced, y_train_reduced)


x_test_reduced = dummies_val_set[significant_predictors]

y_test_reduced = dummies_val_set['Competitive?']


y_pred_reduced=fit_reduced.predict(x_test_reduced)


score_reduced = fit_reduced.score(x_test_reduced, y_test_reduced)
print('\nAccuracy of reduced model : ',score_reduced)


cnf_matrix2 = metrics.confusion_matrix(y_test_reduced, y_pred_reduced)
print('\nConfusion Matrix of reduced model : \n',cnf_matrix2)

print('\nExpected Variance : ',df['Competitive?'].var())

print('\nObserved Variance : ',y_pred_reduced.var())



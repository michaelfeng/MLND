import random
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
import csv

#names = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
#names = ['Survived']

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv")#, sep=",",names=names)#, dtype={"Age": np.float64, "Survived":np.string}, )
test = pd.read_csv("../input/test.csv")#, names=names)#, dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)

print '### Clean and Preprocess train data ###'
train_features = train.loc[:,('Survived','Pclass','Sex', 'Age', 'SibSp','Parch','Fare','Embarked')]
train_features = train_features[train_features.Embarked.notnull()]
train_features = train_features.fillna(random.random() * 60)

train_labels = train_features.pop('Survived')
print len(train_features)
print len(train_labels)


train_features['Gender'] = train_features['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
del train_features['Sex']
print '### Clean and Preprocess test data  ###'

test_features = test.loc[:,('PassengerId','Pclass','Sex', 'Age', 'SibSp','Parch','Fare','Embarked')]
test_features = test_features.fillna(random.random() * 60)
id_test_features = test_features.loc[:,('PassengerId')]
del test_features['PassengerId']

test_features['Gender'] = test_features['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
del test_features['Sex']



# Use Random forest algorithm with GridSearchCV, best estimators--> {'n_estimators': 30} 
'''
param_test1 = {'n_estimators':range(10,101,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10),param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(train_features, train_labels)
print '# best max_depth and min_samples_split: '
print gsearch1.grid_scores_
print gsearch1.best_params_
print gsearch1.best_score_
'''

# best min_sample_split, max_depth --> {'min_samples_split': 50, 'max_depth': 5}
'''
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 80,min_samples_leaf=50,max_features='sqrt' ,oob_score=True, random_state=10),param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(train_features, train_labels)
print '# best max_depth and min_samples_split: '
print gsearch2.grid_scores_
print gsearch2.best_params_
print gsearch2.best_score_
'''

#rf1 = RandomForestClassifier(n_estimators=30, max_depth=5, min_samples_split=50,max_features='sqrt' ,oob_score=True, random_state=10)
#rf1 = RandomForestClassifier(n_estimators= 60, max_depth=5, min_samples_split=50,min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10)
#rf1.fit(train_features, train_labels)
#print rf1.oob_score_


# min_samples_leaf --> {'min_samples_leaf': 6}
'''
param_test3 = {'min_samples_leaf':range(1,100,1)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 80,min_samples_leaf=4,max_depth=11,max_features='sqrt' ,oob_score=True, random_state=10),param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(train_features, train_labels)
print '# Best min_samples_leaf: '
print gsearch3.grid_scores_
print gsearch3.best_params_
print gsearch3.best_score_
'''

# Random_state : 87
'''
param_test4 = {'random_state':range(1,100,1)}
gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=80,min_samples_leaf=4,max_depth=11, max_features='sqrt' ,oob_score=True, random_state=10),param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(train_features, train_labels)
print '# Best random_state:'
print gsearch4.grid_scores_
print gsearch4.best_params_
print gsearch4.best_score_
'''


random_forest = RandomForestClassifier(n_estimators=80,max_depth=7,min_samples_split=110,min_samples_leaf=3, random_state=48, max_features='sqrt', oob_score=True)
random_forest.fit(train_features, train_labels)
pred = random_forest.predict(test_features)
print len(test_features)
print len(pred)

result =  zip(id_test_features,pred)
print type(result)

submission = pd.DataFrame({ 'PassengerId': id_test_features,'Survived': pred })
submission.to_csv("submission.csv", index=False)
print random_forest.score(train_features, train_labels)


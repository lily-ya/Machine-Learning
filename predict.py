import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import csv
# from scipy import stats

#read all data from csv file
df = pd.read_csv('data.csv')
# data.iloc[0] #show the first line

#select feature columns that are relavent to the model
feature_names = ['age', 'male', 'friend_cnt', 'avg_friend_age', 'avg_friend_male', 'friend_country_cnt', 'subscriber_friend_cnt', 'songsListened', 'lovedTracks', 'posts', 'playlists', 'shouts', 'tenure', 'good_country']
features = df[feature_names]

#select target column
target_name = ['adopter']
target = df[target_name]
## features = data_relevent.drop(['adopter'], axis=1)

# add a column Actual_Result
result_names = ["free", "upgrade"]
# df["Actual_Result"] = pd.Categorical.from_codes(target.values, result_names)

ratio = 0.3  #ratio used to split the data
seed = 12    #starting point of random number generator

#split the data to training set and test set
train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=ratio, random_state=seed)

#split the training data to training set and validation set because the method of nearest neighbors will be used to oversample the data
train_x, validate_x, train_y, validate_y = train_test_split(train_features, train_target, test_size=ratio, random_state=seed)

##resample the training data use the SMOTE svm method, SMOTE algorithm uses the nearest neighbors of observations to create synthetic data
# method = SMOTE(ratio={0:12851, 1:12851}, random_state=seed, kind='svm')
# x, y = method.fit_sample(train_x, train_y.values.ravel())
sm = SMOTEENN(random_state=seed, ratio={0:12851, 1:25000})
x, y = sm.fit_sample(train_x, train_y.values.ravel())

print(np.bincount(train_y.values.ravel())) #original count
print(np.bincount(y)) #after oversampling

# stats.describe(y)

#train the model use Random Forest Classifier, default n_estimators=10
model_rf = RandomForestClassifier(random_state=seed, max_features=3, n_estimators=100, class_weight='balanced')
model_rf.fit(x, y)

#confusion matrix
preds_result = model_rf.predict(test_features)
# pred_result = pd.Categorical.from_codes(preds_result, result_names)
matrix = pd.crosstab(index=test_target['adopter'], columns=preds_result, rownames=['actual'], colnames=['preds'])
print(matrix)
matrix.to_excel('matrix.xlsx', index=False)


#get feature importance
importances = model_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# save the feature importance(Mean decrease impurity) to excel
entries = [['feature_id', 'feature_names', 'feature_importances']]
i = 0
for i in range(len(importances)):
	entry = []
	entry.append(i+1)
	entry.append(feature_names[i])
	entry.append(importances[i])
	entries.append(entry)

with open('feature_importances.csv', 'w') as output:
    writer = csv.writer(output, delimiter= ',', lineterminator = '\n')
    writer.writerows(entries)

# Plot the feature importances 
plt.figure(1)
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="b", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), features.columns[indices], rotation=90)
plt.ylim(ymin=0)
plt.ylabel('Relative Importance')
plt.tight_layout()
plt.savefig('importance.png')

##example of horizontal bars
# plt.figure(1)
# plt.title('Feature Importances')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), features.columns[indices])
# plt.xlabel('Relative Importance')
# plt.show()

#save the prediction probabilities to excel
results = [('id', 'actual_result', 'preds_result', 'predict_proba')]
proba = model_rf.predict_proba(test_features)
results.extend(list(zip(test_target['adopter'].index, test_target['adopter'].values, preds_result, np.array(proba)[:,1])))

with open('prediction_results.csv', 'w') as output:
    writer = csv.writer(output)
    writer.writerows(results)
#print the Test results of the Random Forest Model
print('\nTest Results')
print(model_rf.score(test_features, test_target))
print(recall_score(test_target, model_rf.predict(test_features)))



# print('\nValidation Results')
# print(model_rf.score(validate_x, validate_y))
# print(recall_score(validate_y, model_rf.predict(validate_x)))



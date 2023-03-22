from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay

# loading the dataset
heart_dataset_path = 'heart_2020_cleaned.csv'
heart = pd.read_csv(heart_dataset_path)

# visualizing
fig = plt.figure(figsize=(24, 10))
i = 0
for column in heart:
    plt.subplot(3, 6, i+1)
    plt.xlabel(column, fontsize=6)
    plt.tick_params(axis='both', labelsize=6)
    heart[column].value_counts().plot(kind='bar')
    plt.locator_params(axis='x', nbins=15)
    i += 1
plt.subplot_tool()



# preparing
full_pipeline = ColumnTransformer(transformers=[
    ('numpipe', StandardScaler(), make_column_selector(dtype_include=np.number)
     ), ('catpipe', OrdinalEncoder(), make_column_selector(dtype_include=object))
])
heart_prepared = full_pipeline.fit_transform(heart)
col = [heart.select_dtypes(include=['float64']).columns.tolist()+heart.select_dtypes(include=['object']).columns.tolist()]
df_prep = pd.DataFrame(heart_prepared, columns=col)

# checking if correlations exist
corr_matrix = df_prep.corr()
corrs = corr_matrix['HeartDisease'].squeeze().sort_values(ascending=False)

# getting rid of unneeded features
newprepared = df_prep.drop(
    ['Asthma', 'Race', 'MentalHealth', 'SleepTime'], axis=1)

# Visualizing the prepared data to understand it better
newprepared.plot(kind='scatter', x='BMI', y='Stroke', s=newprepared['AgeCategory'], alpha=0.4, c='HeartDisease', cmap='viridis')

# removing some instances from the 'No' category in the 'HeartDisease' target feature to
#balance the dataset
# (startegy called undersampling )
condition = newprepared['HeartDisease'] == 0
newprepared.drop(newprepared[condition.values]
                 [:120000].index, axis=0, inplace=True)

# splitting the data into the train and test sets
x = newprepared.drop("HeartDisease", axis=1)
y = newprepared["HeartDisease"].copy()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10,
                                                    random_state=42, stratify=y)

# fine tuning the random forset classifier
rfc = RandomForestClassifier(min_samples_leaf=5, min_samples_split=4, n_estimators=30, n_jobs=-1, class_weight={0: 1, 1: 50})
rf = GridSearchCV(rfc,[{'n_estimators': [10, 20, 30], 'min_samples_leaf':[1, 5, 10]}, {'min_samples_split': [4, 10], 'n_estimators':[10,20,30],'min_samples_leaf':[1, 5, 10]}],scoring='balanced_accuracy', cv=4, n_jobs=-1)
rf.fit(X_train, y_train)
print('best parameters for random forest: '+str(rf.best_params_))
print('best mean test score for random forest:'+str(max(rf.cv_results_['mean_test_score'])))

# fine tuning the stochastic gradient descent classifier
sgdc=SGDClassifier(loss='log', penalty='elasticnet',
                       class_weight={0: 1, 1: 20})
sg=GridSearchCV(sgdc,[{'loss': ['hinge', 'modified_huber', 'log']}, {'loss': ['hinge', 'modified_huber', 'log'],'penalty': ['elasticnet']}, {'loss': ['hinge', 'modified_huber', 'log'], 'penalty': ['elasticnet'],'alpha':[0.1]}],scoring = 'balanced_accuracy', cv = 4, n_jobs = -1)
sg.fit(X_train, y_train)
print('best parameters for sgd: '+str(sg.best_params_))
print('best mean test score for sgd: ' +
str(max(sg.cv_results_['mean_test_score'])))

# Building a voting classifier and evaluating it on the test set
VC=VotingClassifier([('stochastic gradient', sgdc), ('randomforest',rfc)],voting='soft', n_jobs=-1)
VC.fit(X_train, y_train)
predictions = VC.predict(X_test)
print('accuracy score= '+str(balanced_accuracy_score(y_test, predictions)))

# printing the confusion matrix and f1 score
print(confusion_matrix(y_test, predictions))
print('f1 score= '+str(f1_score(y_test, predictions)))
ConfusionMatrixDisplay(VC, X_test, y_test)
plt.show()
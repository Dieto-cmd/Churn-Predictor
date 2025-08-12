import numpy as np
import google.colab
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import seaborn as sns
from sklearn.model_selection import GridSearchCV

google.colab.drive.mount('/content/drive/')

path_to_folder = "/content/drive/MyDrive/ADiRW/Projekt"

get_ipython().run_line_magic('cd', '$path_to_folder')

get_ipython().system(' ls')



dataset = pd.read_csv('./dataset.csv')
display(dataset.head(5))




queries = pd.read_csv('./queries.csv')
display(queries.head(5))




dataset.isnull().sum()


dataset_processed = dataset.copy()
categorical_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod']
dataset_processed[categorical_columns] = dataset_processed[categorical_columns].fillna('X')


imputer = KNNImputer(n_neighbors= 5)

dataset_processed['tenure'] = imputer.fit_transform(dataset[['tenure']])
dataset_processed['MonthlyCharges'] = imputer.fit_transform(dataset[['MonthlyCharges']])


dataset_processed['TotalCharges'] = pd.to_numeric(dataset_processed['TotalCharges'], errors='coerce')
dataset_processed['TotalCharges'] = dataset_processed['TotalCharges'].fillna(dataset_processed['TotalCharges'].mean())



prob_male = dataset_processed['gender'].value_counts(normalize = True).get('Male',0)
prob_female = dataset_processed['gender'].value_counts(normalize = True).get('Female',0)
dataset_processed['gender'] = dataset_processed['gender'].fillna(np.random.choice(['Male', 'Female'], p=[prob_male, prob_female]))

prob_yes = dataset_processed['PhoneService'].value_counts(normalize = True).get('Yes',0)
prob_no = dataset_processed['PhoneService'].value_counts(normalize = True).get('No',0)
dataset_processed['PhoneService'] = dataset_processed['PhoneService'].fillna(np.random.choice(['Yes', 'No'], p=[prob_yes, prob_no]))

prob_yes = dataset_processed['PaperlessBilling'].value_counts(normalize = True).get('Yes',0)
prob_no = dataset_processed['PaperlessBilling'].value_counts(normalize = True).get('No',0)
dataset_processed['PaperlessBilling'] = dataset_processed['PaperlessBilling'].fillna(np.random.choice(['Yes', 'No'], p=[prob_yes, prob_no]))

prob_yes = dataset_processed['Partner'].value_counts(normalize = True).get('Yes',0)
prob_no = dataset_processed['Partner'].value_counts(normalize = True).get('No',0)
dataset_processed['Partner'] = dataset_processed['Partner'].fillna(np.random.choice(['Yes', 'No'], p=[prob_yes, prob_no]))

prob_yes = dataset_processed['Dependents'].value_counts(normalize = True).get('Yes',0)
prob_no = dataset_processed['Dependents'].value_counts(normalize = True).get('No',0)
dataset_processed['Dependents'] = dataset_processed['Dependents'].fillna(np.random.choice(['Yes', 'No'], p=[prob_yes, prob_no]))

prob_yes = dataset_processed['SeniorCitizen'].value_counts(normalize = True).get(1,0)
prob_no = dataset_processed['SeniorCitizen'].value_counts(normalize = True).get(0,0)
dataset_processed['SeniorCitizen'] = dataset_processed['SeniorCitizen'].fillna(np.random.choice([1, 0], p=[prob_yes, prob_no]))
dataset_processed.isnull().sum()




dataset_processed['gender'] = dataset_processed['gender'].map({'Female': 0, 'Male': 1})
dataset_processed['PhoneService'] = dataset_processed['PhoneService'].map({'No': 0, 'Yes': 1})
dataset_processed['PaperlessBilling'] = dataset_processed['PaperlessBilling'].map({'No': 0, 'Yes': 1})
dataset_processed['Partner'] = dataset_processed['Partner'].map({'No': 0, 'Yes': 1})
dataset_processed['Dependents'] = dataset_processed['Dependents'].map({'No': 0, 'Yes': 1})


dataset_processed = pd.get_dummies(dataset_processed, columns=['MultipleLines', 'InternetService', 'OnlineSecurity',
                                                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                                               'StreamingTV', 'StreamingMovies', 'Contract',
                                                               'PaymentMethod'],dtype=int )




dataset_processed['tenure'] = (dataset_processed["tenure"] - dataset_processed["tenure"].min()) / (dataset_processed["tenure"].max() - dataset_processed["tenure"].min())
dataset_processed['MonthlyCharges'] = (dataset_processed["MonthlyCharges"] - dataset_processed["MonthlyCharges"].min()) / (dataset_processed["MonthlyCharges"].max() - dataset_processed["MonthlyCharges"].min())


dataset_processed['TotalCharges'] = (dataset_processed["TotalCharges"] - dataset_processed["TotalCharges"].min()) / (dataset_processed["TotalCharges"].max() - dataset_processed["TotalCharges"].min())




labels = dataset_processed['Churn'].copy()
dataset_processed = dataset_processed.drop('Churn', axis=1)
np.shape(dataset_processed)



g  = sns.catplot(x="MultipleLines", y="Churn", data=dataset, kind="bar", palette="muted", hue="MultipleLines", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="gender", y="Churn", data=dataset, kind="bar", palette="muted", hue="gender", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="SeniorCitizen", y="Churn", data=dataset, kind="bar", palette="muted", hue="SeniorCitizen", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="Partner", y="Churn", data=dataset, kind="bar", palette="muted", hue="Partner", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="Dependents", y="Churn", data=dataset, kind="bar", palette="muted", hue="Dependents", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="PhoneService", y="Churn", data=dataset, kind="bar", palette="muted", hue="PhoneService", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="InternetService", y="Churn", data=dataset, kind="bar", palette="muted", hue="InternetService", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="OnlineSecurity", y="Churn", data=dataset, kind="bar", palette="muted", hue="OnlineSecurity", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="OnlineBackup", y="Churn", data=dataset, kind="bar", palette="muted", hue="OnlineBackup", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="DeviceProtection", y="Churn", data=dataset, kind="bar", palette="muted", hue="DeviceProtection", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="TechSupport", y="Churn", data=dataset, kind="bar", palette="muted", hue="TechSupport", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="StreamingTV", y="Churn", data=dataset, kind="bar", palette="muted", hue="StreamingTV", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="StreamingMovies", y="Churn", data=dataset, kind="bar", palette="muted", hue="StreamingMovies", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()


g  = sns.catplot(x="Contract", y="Churn", data=dataset, kind="bar", palette="muted", hue="Contract", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()



g  = sns.catplot(x="PaperlessBilling", y="Churn", data=dataset, kind="bar", palette="muted", hue="PaperlessBilling", legend=False)
g = g.set_ylabels("Churn probability")
plt.show()



g  = sns.catplot(x="PaymentMethod", y="Churn", data=dataset, kind="bar", palette="muted", hue="PaymentMethod", legend=False)
g = g.set_ylabels("Churn probability")
g.set_xticklabels(rotation=45)
plt.show()



fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))
axes[0].set_title('All')
sns.histplot(dataset['tenure'], kde=True, ax=axes[0], color='cornflowerblue')

axes[1].set_title('Churn')
sns.histplot(dataset[dataset['Churn'] == 1]['tenure'], kde=True, ax=axes[1], color='g')

axes[2].set_title('No Churn')
sns.histplot(dataset[dataset['Churn'] == 0]['tenure'], kde=True, ax=axes[2], color='r')

fig.show()


fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))
axes[0].set_title('All')
sns.histplot(dataset['MonthlyCharges'], kde=True, ax=axes[0], color='cornflowerblue')

axes[1].set_title('Churn')
sns.histplot(dataset[dataset['Churn'] == 1]['MonthlyCharges'], kde=True, ax=axes[1], color='g')

axes[2].set_title('No Churn')
sns.histplot(dataset[dataset['Churn'] == 0]['MonthlyCharges'], kde=True, ax=axes[2], color='r')

fig.show()


dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'], errors='coerce')
#TotalCharges column needs to be a float but it's a string so I convert it with pd.to_numeric()
fig, axes = plt.subplots(1, 3, sharex=True, figsize=(15,5))
axes[0].set_title('All')
sns.histplot(dataset['TotalCharges'], kde=True, ax=axes[0], color='cornflowerblue')

axes[1].set_title('Churn')
sns.histplot(dataset[dataset['Churn'] == 1]['TotalCharges'], kde=True, ax=axes[1], color='g')

axes[2].set_title('No Churn')
sns.histplot(dataset[dataset['Churn'] == 0]['TotalCharges'], kde=True, ax=axes[2], color='r')

fig.show()


PCA_2D = PCA(n_components=2).fit_transform(dataset_processed)
plt.scatter(PCA_2D[:, 0], PCA_2D[:, 1], c=labels, cmap='viridis')
plt.xlabel('Principal Component 1(Before feature Extraction)')


dataset_processed = dataset_processed.drop(['TotalCharges','PhoneService','MultipleLines_No','MultipleLines_No phone service',
                                            'MultipleLines_X','MultipleLines_Yes','gender'], axis=1)
PCA_2D = PCA(n_components=2).fit_transform(dataset_processed)
plt.scatter(PCA_2D[:, 0], PCA_2D[:, 1], c=labels, cmap='viridis')
plt.xlabel('Principal Component 1(After Feature Extraction)')


X_train, X_test, y_train, y_test = train_test_split(dataset_processed, labels, test_size=0.2, random_state=23)


names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=100, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
for name, clf in zip(names, classifiers):
    print(name + " score: ", clf.fit(X_train, y_train).score(X_test, y_test))


AdaBst = AdaBoostClassifier(random_state=42,estimator = DecisionTreeClassifier(random_state=42,max_depth=1))
params = {
    'n_estimators': [1, 5, 10, 20, 50, 100, 200, 500],
    'learning_rate': [0.0001, 0.5, 1, 2, 5]
}
grid_search = GridSearchCV(estimator=AdaBst,param_grid = params, verbose=3, scoring='roc_auc', return_train_score=False, cv=5)
grid_search.fit(dataset_processed, labels)
print("Best score:", grid_search.best_score_)
print("Best hyperparameters:", grid_search.best_params_)


mlp = MLPClassifier(random_state=42,max_iter=600)
params ={'solver': ['adam','sgd'],
         'hidden_layer_sizes' : [[200,],[200,50],[200,150],[200,150,50]],
         'activation': ['relu','identity', 'logistic']}
grid_search = GridSearchCV(estimator=mlp,param_grid = params, verbose=3, scoring='roc_auc', return_train_score=False, cv=5)
grid_search.fit(dataset_processed, labels)
print("Best score:", grid_search.best_score_)
print("Best hyperparameters:", grid_search.best_params_)


SVM = SVC(random_state=42,probability = True,cache_size = 1000,kernel = 'linear')
params ={'C' : [0.0001,0.5,1],}

grid_search = GridSearchCV(estimator=SVM,param_grid = params, verbose=3, scoring='roc_auc', return_train_score=False, cv=5)
grid_search.fit(dataset_processed, labels)
print("Best score:", grid_search.best_score_)
print("Best hyperparameters:", grid_search.best_params_)


SVM = SVC(random_state=42,probability = True,cache_size = 1000,kernel = 'poly')
params ={'C' : [0.0001,0.5,1],
         'degree': [3,4,5],
          'coef0': [0,0.5,1],
         'gamma':['scale','auto']}

grid_search = GridSearchCV(estimator=SVM,param_grid = params, verbose=3, scoring='roc_auc', return_train_score=False, cv=5)
grid_search.fit(dataset_processed, labels)
print("Best score:", grid_search.best_score_)
print("Best hyperparameters:", grid_search.best_params_)



SVM = SVC(random_state=42,probability = True,cache_size = 1000,kernel = 'rbf')
params ={'C' : [0.0001,0.5,1],
         'gamma':['scale','auto']}

grid_search = GridSearchCV(estimator=SVM,param_grid = params, verbose=3, scoring='roc_auc', return_train_score=False, cv=5)
grid_search.fit(dataset_processed, labels)
print("Best score:", grid_search.best_score_)
print("Best hyperparameters:", grid_search.best_params_)



SVM = SVC(random_state=42,probability = True,cache_size = 1000,kernel = 'sigmoid')
params ={'C' : [0.0001,0.5,1],
          'coef0': [0,0.5,1],
         'gamma':['scale','auto']}

grid_search = GridSearchCV(estimator=SVM,param_grid = params, verbose=3, scoring='roc_auc', return_train_score=False, cv=5)
grid_search.fit(dataset_processed, labels)
print("Best score:", grid_search.best_score_)
print("Best hyperparameters:", grid_search.best_params_)


queries_processed = queries.copy()
categorical_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod']
queries_processed[categorical_columns] = queries_processed[categorical_columns].fillna('X')

imputer = KNNImputer(n_neighbors=5)

queries_processed['tenure'] = imputer.fit_transform(queries[['tenure']])
queries_processed['MonthlyCharges'] = imputer.fit_transform(queries[['MonthlyCharges']])

queries_processed = pd.get_dummies(queries_processed, columns=['MultipleLines', 'InternetService', 'OnlineSecurity',
                                                                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                                                'StreamingTV', 'StreamingMovies', 'Contract',
                                                                'PaymentMethod'], dtype=int)

queries_processed['gender'] = queries_processed['gender'].map({'Female': 0, 'Male': 1})
queries_processed['PhoneService'] = queries_processed['PhoneService'].map({'No': 0, 'Yes': 1})
queries_processed['PaperlessBilling'] = queries_processed['PaperlessBilling'].map({'No': 0, 'Yes': 1})
queries_processed['Partner'] = queries_processed['Partner'].map({'No': 0, 'Yes': 1})
queries_processed['Dependents'] = queries_processed['Dependents'].map({'No': 0, 'Yes': 1})

queries_processed['tenure'] = (queries_processed["tenure"] - queries_processed["tenure"].min()) / (queries_processed["tenure"].max() - queries_processed["tenure"].min())
queries_processed['MonthlyCharges'] = (queries_processed["MonthlyCharges"] - queries_processed["MonthlyCharges"].min()) / (queries_processed["MonthlyCharges"].max() - queries_processed["MonthlyCharges"].min())

queries_processed['TotalCharges'] = pd.to_numeric(queries_processed['TotalCharges'], errors='coerce')
queries_processed['TotalCharges'] = queries_processed['TotalCharges'].fillna(queries_processed['TotalCharges'].mean())
queries_processed['TotalCharges'] = (queries_processed["TotalCharges"] - queries_processed["TotalCharges"].min()) / (queries_processed["TotalCharges"].max() - queries_processed["TotalCharges"].min())

prob_male = queries_processed['gender'].value_counts(normalize=True).get(1, 0)
prob_female = queries_processed['gender'].value_counts(normalize=True).get(0, 0)
queries_processed['gender'] = queries_processed['gender'].fillna(np.random.choice([0, 1], p=[prob_female, prob_male]))

prob_yes = queries_processed['PhoneService'].value_counts(normalize=True).get(1, 0)
prob_no = queries_processed['PhoneService'].value_counts(normalize=True).get(0, 0)
queries_processed['PhoneService'] = queries_processed['PhoneService'].fillna(np.random.choice([0, 1], p=[prob_no, prob_yes]))

prob_yes = queries_processed['PaperlessBilling'].value_counts(normalize=True).get(1, 0)
prob_no = queries_processed['PaperlessBilling'].value_counts(normalize=True).get(0, 0)
queries_processed['PaperlessBilling'] = queries_processed['PaperlessBilling'].fillna(np.random.choice([0, 1], p=[prob_no, prob_yes]))

prob_yes = queries_processed['Partner'].value_counts(normalize=True).get(1, 0)
prob_no = queries_processed['Partner'].value_counts(normalize=True).get(0, 0)
queries_processed['Partner'] = queries_processed['Partner'].fillna(np.random.choice([0, 1], p=[prob_no, prob_yes]))

prob_yes = queries_processed['Dependents'].value_counts(normalize=True).get(1, 0)
prob_no = queries_processed['Dependents'].value_counts(normalize=True).get(0, 0)
queries_processed['Dependents'] = queries_processed['Dependents'].fillna(np.random.choice([0, 1], p=[prob_no, prob_yes]))

prob_yes = queries_processed['SeniorCitizen'].value_counts(normalize=True).get(1, 0)
prob_no = queries_processed['SeniorCitizen'].value_counts(normalize=True).get(0, 0)
queries_processed['SeniorCitizen'] = queries_processed['SeniorCitizen'].fillna(np.random.choice([0, 1], p=[prob_no, prob_yes]))

queries_processed = queries_processed.drop(['TotalCharges','PhoneService','MultipleLines_No','MultipleLines_No phone service',
                                            'MultipleLines_X','MultipleLines_Yes','gender'], axis=1)


Classifier = AdaBoostClassifier(random_state=42,learning_rate = 0.5,n_estimators = 200,estimator =DecisionTreeClassifier(random_state=42,max_depth=1)).fit(dataset_processed, dataset['Churn'])


test_predictions = pd.DataFrame(
    dict(
        Score=Classifier.predict_proba(queries_processed)[:,1],
        Label=Classifier.predict(queries_processed)
        )
    )



display(test_predictions.head(50))


test_predictions.to_csv('./test_predictions_Nikakda.csv')



"""
@author Wildo Monges
This program resolves the Titanic problem, competition posted by Kaggle
Note:
    Use python 3.5
    Execute running from terminal python titanic.py
Result:
    In the terminal will display passengerId and Survived columns
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Load training and testing datasets
titanic_train = pd.read_csv('train.csv', index_col=None)
titanic_test = pd.read_csv('test.csv', index_col=None)

# Check the head and the mean to interpret the data
print(titanic_train.head())
print('Group by Survived')
print(titanic_train['Survived'].mean())
print('Group by Pclass')
print(titanic_train.groupby('Pclass').mean())
print('Group by Pclass and Age')
print(titanic_train.groupby(['Pclass', 'Sex']).mean())
print('Group by Age, take the average of the age')
age_avg = round(titanic_train['Age'].mean(), 0)
print(age_avg)

print(titanic_train.count())

# Check if we have the same amount of rows for each column
print(titanic_train.count())


# Function to preproces the dataset read
def preprocess_titanic_data(data):
    processed_data = data.copy()
    encoder = preprocessing.LabelEncoder()
    processed_data.Sex = encoder.fit_transform(processed_data.Sex)
    processed_data.Embarked = encoder.fit_transform(processed_data.Embarked.fillna('0'))
    processed_data = processed_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    embarket_avg = round(processed_data.Embarked.mean(), 0)
    processed_data.Embarked = processed_data.Embarked.fillna(embarket_avg)

    processed_data.Age = processed_data.Age.fillna(age_avg)

    fare_avg = round(processed_data.Fare.mean(), 3)
    processed_data.Fare = processed_data.Fare.fillna(fare_avg)

    return processed_data


processed_data_train = preprocess_titanic_data(titanic_train)
# Counting
print(processed_data_train.count())

# Get inputs(X) and outputs(y)
y = processed_data_train['Survived'].values
X = processed_data_train.drop(['Survived'], axis=1).values

# Run a cross validation 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print('X Train')
print(X_train)
print('y train')
print(y_train)
# Compare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RFC', RandomForestClassifier()))
models.append(('ABC', AdaBoostClassifier()))
models.append(('GBC', GradientBoostingClassifier()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    msg = "%s: MS (%f), AS(%f) " % (name, score, accuracy_score(y_test, predictions))
    print(msg)

# prepare test data
titanic_validation = preprocess_titanic_data(titanic_test)
print("Check the amount of data by column")
print(titanic_validation.count())

# Select GBC because it has the better accuracy score
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
predictions = classifier.predict(titanic_validation)

titanic_validation = titanic_validation.as_matrix(columns=['PassengerId'])
result = [['Survived', 'PassengerId']]
row = 0
passenger_index = 0
stop = len(titanic_validation)
while row < stop:
    result.append([predictions[row], titanic_validation[row][passenger_index]])
    row = row + 1

print('Printing result')
for row in range(len(result)):
    print("%s \t\t\t %s" % (result[row][0], result[row][1]))

print('Done!')

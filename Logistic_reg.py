import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score


# ---------------- Data collection --------------------

# Load Titanic data
titanic_data = pd.read_csv('./titanic.csv')

# Map numerical Pclass values to labels
# class_mapping = {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}
# titanic_data['Pclass'] = titanic_data['Pclass'].map(class_mapping)

# Check data types
# print(titanic_data.dtypes)

# Convert necessary columns to categorical
titanic_data['Survived'] = titanic_data['Survived'].astype('category')

# ---------------- Data analysis --------------------

# Create the countplot
# sns.countplot(x="Survived", hue="Sex", data=titanic_data) # relation of survived with gender 
# sns.countplot(x="Survived", hue="Pclass", data=titanic_data) 

# plt.hist(titanic_data['Age'], bins=10, color='black') 
# plt.show() 


# ------------------ Data Wrangling (Cleaning the NULL or unnecessary values) ------------------------

# sns.boxplot(x='Pclass', y='Age', data=titanic_data)

titanic_data = titanic_data.drop('Cabin', axis=1)
titanic_data = titanic_data.drop('Ticket', axis=1)
titanic_data = titanic_data.drop('Fare', axis=1)
titanic_data = titanic_data.drop('PassengerId', axis=1)
titanic_data = titanic_data.drop('Name', axis=1)
titanic_data = titanic_data.dropna() # dropping all the rows that had one or more nulls
# heat map 
# sns.heatmap(titanic_data.isnull(), yticklabels=False) # getting to see where we have null values in graph view 

# now we are going to assign dummy categrical values to columns with strings because logistic regression accepts only categories not strings or numbers e.g. for sex = male/female, we will make a male column 0 or 1 (not a male or male)

sex = pd.get_dummies(titanic_data['Sex'], drop_first=True) # it made two columns female and male and dropped female 
embark = pd.get_dummies(titanic_data['Embarked']) # S, Q, C columns 
Pclass = pd.get_dummies(titanic_data['Pclass'], prefix='class') # 1, 2, 3

# adding these columns to original data

titanic_data = pd.concat([titanic_data, sex, embark, Pclass], axis=1)
titanic_data = titanic_data.drop('Sex',  axis=1)
titanic_data = titanic_data.drop('Pclass',axis=1)
titanic_data = titanic_data.drop('Embarked', axis=1)
# print(titanic_data.head(4))


# -------------- Training and testing the model ------------------------

logistic = LogisticRegression()

x = titanic_data.drop('Survived', axis=1) # every column except the survived
y = titanic_data['Survived'] # output 

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.05, random_state=12)
logistic.fit(x_train, y_train)

predictions = logistic.predict(x_test)


# ------------------- Accuracy -----------------------------

classificationReport = classification_report(y_test, predictions)
confusionMatrix = confusion_matrix(y_test, predictions)

# confusion matrix is of form 
#               Predicted No | Predicted Yes 
# Actual No  |       72      |      14
# Actual Yes |       18      |      39
#
print(classificationReport)
print(confusionMatrix)
print(accuracy_score(y_test, predictions)*100) # accuracy percentage 
plt.show()
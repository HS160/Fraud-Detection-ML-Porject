import numpy as np 
import pandas as pd 
import seaborn as sns 
import  matplotlib.pyplot as plt 

#matplotlib inline
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression 

sns.set_style('darkgrid')

## Dataset 
df = pd.read_csv('payment_fraud.csv')

df.head()

df.isnull().sum() ## checking the null valeus 

paymthd = df.paymentMethod.value_counts()
plt.figure(figsize=(5, 5))
sns.barplot(x=paymthd.index, y=paymthd)
plt.ylabel('Count')
plt.title('Payment Method Distribution')
plt.xticks(rotation=45)  # Optional: rotate x-axis labels if they're long
plt.tight_layout()  # Adjust layout to prevent cutting off labels
plt.show()

df.label.value_counts() ## count the number of 0's and 1's

## coverting paymentMethod column into label encoding
paymthd_label = {v:k for k, v in enumerate(df.paymentMethod.unique())}

df.paymentMethod = df.paymentMethod.map(paymthd_label)

df.head()

## corr(): it gives the correlation between the featuers
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True);

df.describe()

## independent and dependent features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

## scaling 

sc = StandardScaler()
X = sc.fit_transform(X)

## train test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

## logisticRegression Model
lg = LogisticRegression()

## training
lg.fit(X_train, y_train)
## prediction 
pred = lg.predict(X_test)

print("----------------------------------------------------Accuracy------------------------------------------------------")
print(accuracy_score(y_test, pred))
print()

print("---------------------------------------------------Classification Report---------------------------------------------")
print(classification_report(y_test, pred))
print()

print("-------------------------------------------------Confustion Metrics----------------------------------------------------")
plt.figure(figsize=(10, 10));
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='g');


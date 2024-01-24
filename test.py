import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# make plots nicer looking
sns.set_style('whitegrid')

# Read in csv
df = pd.read_csv('KNN_Project_Data')

# Check head of the data frame

# print(df.head())

# Exploratory Data Analysis

# sns.pairplot(data = df, hue = 'TARGET CLASS')
# plt.savefig('./plots/pairplot.png')
# plt.show()

# Standardize the Variables

scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS', axis = 1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))

df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])

# check if the features have been scaled properly
print(df_feat.head())

# train test split

X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)

# knn = KNeighborsClassifier(n_neighbors = 1)

# knn.fit(X_train, y_train)

# Predictions and Evaluations

# pred = knn.predict(X_test)

# print(confusion_matrix(y_test, pred))
# print(classification_report(y_test, pred))

# Plotting out possible k values

error_rate = []


for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# Create plot for each k value

plt.figure(figsize = (10,6))

plt.plot(range(1,40), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o', markerfacecolor = 'red', markersize = 10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

plt.show()

# To do: Create a function that will pick the k-value with the least amount of error in this model

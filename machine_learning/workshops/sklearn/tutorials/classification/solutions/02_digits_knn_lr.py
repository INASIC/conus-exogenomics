""" 1. Load the data """

from sklearn.datasets import load_digits

digits = load_digits()


""" 2. Assign features and targets"""

X = digits.data
y = digits.target


""" 3. Seperate data into training set and testing set """

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.33,
                                                   random_state=1234)

""" 4. Train your models"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

lr = LogisticRegression()
lr.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


""" 5. Score their accuracy """

print('Logistic Regression accuracy:', lr.score(X_test, y_test))
print('k-Nearest Neighbours accuracy:', knn.score(X_test, y_test))

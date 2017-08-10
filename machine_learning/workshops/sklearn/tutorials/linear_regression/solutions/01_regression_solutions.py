from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()
knr.fit(X_train, y_train)
print('R^2 =', lr.score(X_test, y_test))

y_pred_train = knr.predict(X_train)

plt.scatter(X_train, y_train, c='b')
plt.scatter(X_train, y_pred_train, c='r')
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from LogisticRegression import sigmoid

df = datasets.load_breast_cancer()
X, y = df.data, df.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

model = LogisticRegression(lr=0.0001,epochs=5000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy: ",model.accuracy(y_pred,y_test)*100)

x_values = np.linspace(-10, 10, 100)
y_values = sigmoid(x_values)
X_values = range(len(y_pred))

plt.scatter(X_values,y_pred)
plt.plot(x_values, y_values, label="Sigmoid Function")
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid(True)
plt.show()

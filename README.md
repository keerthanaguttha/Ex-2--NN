<H3>Name: Guttha Keerthana</H3>
<H3>Register no.: 212223240045</H3>
<H3>Date:20-09-2025</H3>
<H3>Experiment No. 2 </H3>
## Implementation of Perceptron for Binary Classification
# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Write a class for perceptron with fit and predict function with sigmoid activation function
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=100):
        self.lr = learning_rate
        self.n_iter = n_iter
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.errors_ = []
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0

        for _ in range(self.n_iter):
            linear_output = np.dot(X, self.w_) + self.b_
            y_pred = self.sigmoid(linear_output)

            errors = y - y_pred
            self.w_ += self.lr * np.dot(errors, X)
            self.b_ += self.lr * errors.sum()
            
            # Count misclassifications
            y_pred_class = np.where(y_pred >= 0.5, 1, -1)
            self.errors_.append(np.sum(y_pred_class != y))
        return self

    def predict(self, X):
        linear_output = np.dot(X, self.w_) + self.b_
        return np.where(self.sigmoid(linear_output) >= 0.5, 1, -1)


# Start your main here ,read the iris data set
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                 header=None)

# map the labels to a binary integer value
df.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
df = df[df['class'].isin(['Iris-setosa','Iris-versicolor'])]  # Binary classification

X = df[['sepal_length','sepal_width','petal_length','petal_width']].values
y = np.where(df['class'] == 'Iris-setosa', 1, -1)

# standardization of the input features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# train the model by setting the learning rate as 0.01
model = Perceptron(learning_rate=0.01, n_iter=50)
model.fit(X_train, y_train)
plt.figure(figsize=(10,6))
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolour')
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
plt.show()


# plot the number of errors during each iteration
plt.plot(range(1, len(model.errors_)+1), model.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Training Error per Epoch')
plt.show()


# print the accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.4f}")

```
    

# OUTPUT:
<img width="1052" height="658" alt="image" src="https://github.com/user-attachments/assets/3e600942-ef3e-4798-b3c3-a0f01d6f4649" />
<img width="703" height="574" alt="image" src="https://github.com/user-attachments/assets/29bd25d9-fba2-412d-929d-d8f837a09ca9" />
<img width="474" height="62" alt="image" src="https://github.com/user-attachments/assets/e40b359d-2aa9-46d8-b990-ff6a1f0f024c" />




# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 

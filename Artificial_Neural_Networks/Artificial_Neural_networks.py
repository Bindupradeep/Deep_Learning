# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#Splitting the datset into train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# IMporting the Keras Libraries and Packages
import keras
from keras.models import Sequential # To initialize the Neural Network
from keras.layers import Dense # To create the Layers in The Artificial Neural Network
# Dense will take care of the weights Initiallization of Step 1
# The Input Node are the numner of Columns i.e., 11(11 input nodes Step : 2)

# Initializing the ANN
classifier = Sequential()

# Adding the Input Layer and the First Hidden Layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
# Output_dim = 6 => 11+1/2; relu => Recifier Activation Method

#Adding the Second Hidden Layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))

# Adding the Outout Layer
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

# Compiling the ANN
classifier.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test Set Results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


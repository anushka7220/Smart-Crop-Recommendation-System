# importing all the dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
# loading the dataset
data = pd.read_csv("Crop_recommendation.csv")

# Displaying the first few rows of the dataframe
print(data.head())

# Displaying the shape of the dataframe
print(f"Data Shape: {data.shape}")

# Checking for null values in the dataframe
print(f"Null values in the dataset: {data.isnull().sum()}")

# Splitting the features and labels
x = data.iloc[:,:-1]  # features
y = data.iloc[:,-1]   # labels

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initializing and training the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Making predictions on the test set
predictions = model.predict(x_test)
pickle.dump(model, open("model.pickle", "wb"))
# Calculating the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Making a prediction for a new set of features
new_features = [[36, 58, 25, 28.66024, 59.31, 8.39, 36.92]]
predicted_crop = model.predict(new_features)
print("Predicted Crop:", predicted_crop)

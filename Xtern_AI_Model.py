import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_excel('XTern 2024 Artificial Intelegence Data Set.xlsx')

# Select the numeric columns for scaling
numeric_cols = data.select_dtypes(include=[float, int]).columns.tolist()

# Drop the non-numeric columns and the 'Order' column
X = data.drop(['Order'] + data.select_dtypes(exclude=[float, int]).columns.tolist(), axis=1)
Y = data['Order']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the numeric columns using StandardScaler
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train.select_dtypes(include=[float, int]))
X_test[numeric_cols] = scaler.transform(X_test.select_dtypes(include=[float, int]))

# Train a RandomForestClassifier model on the training data
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, Y_train)

# Save the trained model to a file
model_filename = 'XTern_AI_Model.pkl'
joblib.dump(model, model_filename)

# Define a class for making predictions using the trained model
class CustomerOrderPredictor:
    def __init__(self, model_filename):
        self.model = joblib.load(model_filename)
    
    def predict(self, data):
        # Perform any necessary data preprocessing on 'data'
        processed_data = data.drop(data.select_dtypes(exclude=[float, int]).columns.tolist(), axis=1)
        processed_data[numeric_cols] = scaler.transform(processed_data.select_dtypes(include=[float, int]))
        
        # Make predictions using the loaded model
        predictions = self.model.predict(processed_data)
        return predictions

# Create an instance of the CustomerOrderPredictor class
predictor = CustomerOrderPredictor('XTern_AI_Model.pkl')

# Make a prediction on new data
new_data = pd.DataFrame({
    'University': ['Butler University'],
    'Year': ['Year 3'],
    'Major': ['Astronomy'],
    'Time': [20],
})

predictions = predictor.predict(new_data)
print(predictions)
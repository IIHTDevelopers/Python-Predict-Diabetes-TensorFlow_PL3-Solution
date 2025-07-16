import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
def load_dataset():
    local_path = "pima-indians-diabetes.data.csv"
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(local_path, names=col_names)
    return df

# 2. Preprocess dataset
def preprocess_data(df):
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# 3. Build dense model for tabular classification
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# 4. Compile model
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# 5. Train model
def train_model(model, X_train, y_train, X_val, y_val, epochs=20):
    history = model.fit(X_train, y_train, epochs=epochs,
                        validation_data=(X_val, y_val), batch_size=32, verbose=0)
    print("Training completed.")
    return history

# 6. Evaluate model
def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc*100:.2f}%")
    return acc

# 7. Predict on new data sample
def predict_sample(model, sample):
    sample = np.array(sample).reshape(1, -1)
    prediction = model.predict(sample, verbose=0)[0][0]
    return "Diabetic" if prediction >= 0.5 else "Non-Diabetic"

# 8. Prepare sample input
def prepare_sample_input(raw_sample, scaler):
    return scaler.transform([raw_sample])[0]

# 9. Load sample from file
def load_sample_from_file(filename="sample_input.txt"):
    try:
        with open(filename, "r") as f:
            line = f.readline().strip()
            return [float(x.strip()) for x in line.split(",")]
    except Exception as e:
        print("Error reading sample input from file:", e)
        return None

# ===== Main Driver Code =====
if __name__ == "__main__":
    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    model = build_model(input_dim=X_train.shape[1])
    compile_model(model)
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=20)
    evaluate_model(model, X_test, y_test)

    raw_sample = load_sample_from_file("sample_input.txt")
    if raw_sample:
        processed_sample = prepare_sample_input(raw_sample, scaler)
        result = predict_sample(model, processed_sample)
        print("Prediction for sample:", result)
    else:
        print("Prediction could not be performed due to invalid input.")

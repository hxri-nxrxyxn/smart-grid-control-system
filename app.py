from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt
        from tensorflow.keras.callbacks import EarlyStopping

        # Load the datasets
        file_path_energy = './energy_dataset.xlsx'  # Update with your local path
        file_path_weather = './weather_features.xlsx'  # Update with your local path
        energy_data = pd.read_excel(file_path_energy, sheet_name='energy_dataset')
        weather_data = pd.read_excel(file_path_weather)

        # Convert the 'time' column to datetime and set it as the index
        energy_data['time'] = pd.to_datetime(energy_data['time'], utc=True)
        energy_data.set_index('time', inplace=True)
        weather_data['time'] = pd.to_datetime(weather_data['time'], utc=True)
        weather_data.set_index('time', inplace=True)

        # Merge the datasets on the time index
        combined_data = energy_data.merge(weather_data, how='outer', left_index=True, right_index=True)

        # Drop rows with NaN values in the target column 'total load actual'
        combined_data.dropna(subset=['total load actual'], inplace=True)

        # Fill NaN values in 'city_name' with a placeholder
        combined_data['city_name'].fillna('Unknown', inplace=True)

        # Standardize city names by stripping whitespace and converting to lowercase
        combined_data['city_name'] = combined_data['city_name'].str.strip().str.lower()

        # Select relevant features: 'total load actual' and 'city_name'
        # Convert 'city_name' to numeric values using one-hot encoding
        city_encoded = pd.get_dummies(combined_data['city_name'], drop_first=False)

        # Add city encoded features to the combined dataset
        combined_data = pd.concat([combined_data[['total load actual']], city_encoded], axis=1)

        # Feature scaling for load data and city encoded features
        scaler_load = MinMaxScaler(feature_range=(0, 1))
        scaler_city = MinMaxScaler(feature_range=(0, 1))

        # Scale load and city data separately
        scaled_load = scaler_load.fit_transform(combined_data[['total load actual']])
        scaled_city = scaler_city.fit_transform(combined_data[city_encoded.columns])

        # Combine scaled load data with city encoded features
        scaled_data = np.hstack((scaled_load, scaled_city))

        # Create sequences for LSTM, including city features
        sequence_length = 24  # 24 hours (1 day) sequence length
        X = []
        y = []

        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Predicting the load value

        X, y = np.array(X), np.array(y)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build the LSTM model with additional features for city name
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(units=50))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        # Evaluate the model on test data
        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler_load.inverse_transform(y_pred.reshape(-1, 1))
        y_test_rescaled = scaler_load.inverse_transform(y_test.reshape(-1, 1))

        # Plotting the predicted vs actual values
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_rescaled, label='Actual Load')
        plt.plot(y_pred_rescaled, label='Predicted Load')
        plt.title('Predicted vs Actual Load')
        plt.xlabel('Time Steps')
        plt.ylabel('Total Load Actual')
        plt.legend()
        plt.savefig("my_plot.png")

        # Function to predict future load for a specific date
        def predict_future_load(input_date, city_name):
            input_date = pd.to_datetime(input_date, utc=True)
            if input_date <= energy_data.index[-1]:
                raise ValueError("Input date must be in the future.")

            # Standardize the city name input
            city_name = city_name.strip().lower()

            if city_name not in city_encoded.columns:
                raise ValueError("City name not found in the dataset. Available cities: {}".format(', '.join(city_encoded.columns)))

            # Use a loop to predict multiple future points to ensure variability
            future_sequence = scaled_data[-sequence_length:, :].copy()
            city_feature_index = list(city_encoded.columns).index(city_name)
            city_feature_values = np.zeros(len(city_encoded.columns))
            city_feature_values[city_feature_index] = 1
            city_feature_values = city_feature_values.reshape(1, -1)  # Reshape to match future step requirements
            city_feature_values_scaled = scaler_city.transform(pd.DataFrame(city_feature_values, columns=city_encoded.columns))  # Scale the city feature values

            days_to_predict = (input_date - energy_data.index[-1]).days
            if days_to_predict <= 0:
                raise ValueError("Input date must be in the future.")

            for _ in range(days_to_predict):
                # Predict the next load value
                predicted_scaled = model.predict(future_sequence.reshape(1, sequence_length, -1))

                # Ensure city feature values match the expected dimensions
                city_feature_values_scaled = np.tile(city_feature_values_scaled, (predicted_scaled.shape[0], 1))

                # Create the new step with predicted load and city features
                new_step = np.hstack((predicted_scaled, city_feature_values_scaled))

                # Update future sequence by removing the oldest time step and adding the predicted value
                future_sequence = np.vstack((future_sequence[1:], new_step))

            # Rescale the predicted load back to original scale
            predicted_load = scaler_load.inverse_transform(predicted_scaled.reshape(-1, 1))
            return predicted_load[0, 0]
        # User input for predicting future load
        user_input_date = request.form['date']  # User is prompted to input a date
        user_input_city = request.form['text']  # User is prompted to input city name
        try:
            future_load = predict_future_load(user_input_date, user_input_city)
            print(f'Predicted total actual load for {user_input_date} in {user_input_city}: {future_load}')
        except ValueError as e:
            print(e)
        return render_template('output.html', future_load=future_load)

    else:
        return render_template('index.html')

@app.route('/representation')
def representation():
    return render_template('representation.html')

@app.route('/output')
def output():
    return render_template('output.html')

if __name__ == '__main__':
    app.run(debug=True)

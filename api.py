from flask import Flask, jsonify, request, json, Request
import pickle
import numpy as np
import pandas as pd
import preprocessing

kapi = Flask(__name__)


@kapi.route('/predict', methods=['POST'])
def predict():

    # Load the KMeans model from the pickle file
    with open('demographic_segmentation.pkl', 'rb') as kmod:
        model = pickle.load(kmod)

        if request.method == 'POST':
            if request.get_json():
                data = json.loads(request.get_data())

                # # Convert the input data to a dataframe
                print("Damn there is data", data, '----------')
                df = pd.DataFrame([data])
                print("Damn there is data", df,
                      '-------------------------------------------')

                # # preprocess data
                norm_data = preprocessing.preprocess_data(df)

                # # Use the KMeans model to make a prediction
                prediction = model.predict(norm_data)

                # # Create the response data as a dictionary
                response_data = {'prediction': int(prediction)}

                # Return the response data as a JSON response
                return jsonify(response_data)
            else:
                return {"error": "No data provided"}
        else:
            return jsonify({'error': 'Invalid request method'})


if __name__ == '__main__':
    kapi.run(host='127.0.0.1', port=5000)

from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():

    try:
        json_ = request.json
        query = pd.json_normalize(json_)
        res = np.array(query).reshape(1,-1)
        prediction = rf.predict_proba(res)
        risk = prediction[0 ,1]
                
        return jsonify({'risk of diabetes, %': risk*100})

    except:
            return jsonify({'trace': traceback.format_exc()})     



if __name__ == '__main__':

    try:
        port = int(sys.argv[1]) 
    except:
        port = 12345 
    
    rf = joblib.load('random_forest_model_diabetes_refined_31_5_2021.pkl') # Load "model.pkl"
    print ('Model loaded')
    
    app.run(debug=True, port=port)
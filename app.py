from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    try:
        item_weight = float(request.form['item_weight'])
        item_fat_content = float(request.form['item_fat_content'])
        item_visibility = float(request.form['item_visibility'])
        item_type = float(request.form['item_type'])
        item_mrp = float(request.form['item_mrp'])
        outlet_establishment_year = float(request.form['outlet_establishment_year'])
        outlet_size = float(request.form['outlet_size'])
        outlet_location_type = float(request.form['outlet_location_type'])
        outlet_type = float(request.form['outlet_type'])

        X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                       outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

        scaler_path = r'C:\Users\sachin\Desktop\Project Data Anlaytics\BigMart Sales\Models\sc.sav'
        model_path = r'C:\Users\sachin\Desktop\Project Data Anlaytics\BigMart Sales\Models\lr.sav'

        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            return jsonify({'error': 'Model or scaler file not found'}), 500

        sc = joblib.load(scaler_path)
        X_std = sc.transform(X)

        model = joblib.load(model_path)
        Y_pred = model.predict(X_std)

        return jsonify({'Prediction': float(Y_pred[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=9457)

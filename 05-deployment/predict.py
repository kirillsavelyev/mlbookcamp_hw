import pickle
from flask import Flask, request, jsonify

app = Flask('GetCC')

with open('dv.bin', 'rb') as f:
    dv = pickle.load(f)
with open('model2.bin', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])    
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5
    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn)
    }
    return jsonify(result)

if __name__ == "__main__":
    # customer = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
    # print(predict(customer))
    app.run(debug=True, host='0.0.0.0', port=9696)
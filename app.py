from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)


# Load the model from the correct path
model = joblib.load('Anti-Money-Laundering-with-Machine-Learning-main/model/random_forest_model.pkl')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = [[float(data['type']), float(data['amount']), float(data['oldbalanceOrg']),
                   float(data['newbalanceOrig']), float(data['oldbalanceDest']), float(data['newbalanceDest'])]]
    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

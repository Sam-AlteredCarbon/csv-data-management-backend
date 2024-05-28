from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from static.scripts import *
from flask_cors import CORS  # Import CORS
import os
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

transformations = {
    'smooth': lambda data: smooth(data),
    'logarithm': lambda data: logarithm_transform(data),
    'derivative': lambda data: derivative_transform(data),
    'integral': lambda data: integral_transform(data),
    'exponential_smoothing': lambda data: exponential_smoothing(data),
    'dynamic_moment': lambda data: calculate_dynamic_moments_optimized(data, moment_index=0),
    'z_score': lambda data: normalise_data(data, method='standard'),
    'min_max': lambda data: normalise_data(data, method='minmax'),
    'max_abs': lambda data: normalise_data(data, method='maxabs'),
    'robust': lambda data: normalise_data(data, method='robust'),
    'box_cox': lambda data: normalise_data(data, method='boxcox'),
    'yeo_johnson': lambda data: normalise_data(data, method='yeojohnson'),
    'log_norm': lambda data: normalise_data(data, method='log'),
    'polynomial': lambda data: generate_polynomial_features(data),
    'residuals': lambda data: decompose_time_series(data, model='additive', output_key='resid', freq=4),
    'trend': lambda data: decompose_time_series(data, model='additive', output_key='trend', freq=4),
    'seasonal': lambda data: decompose_time_series(data, model='additive', output_key='seasonal', freq=4),

}

transformations_list = {
    'smooth': smooth,
    'logarithm': logarithm_transform,
    'derivative': derivative_transform,
    'integral': integral_transform,
    'exponential_smoothing': exponential_smoothing,
    'dynamic_moment': calculate_dynamic_moments_optimized,
    'normalise':  normalise_data,
    'polynomial':  generate_polynomial_features,
    'decompose':  decompose_time_series,
}

models = {
    'DecisionTree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,

}


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        samples, class_counts = process_csv(file_path)
        print("Class Counts: ",class_counts)
        print("Samples: ",samples[0]["data"][0])
        return jsonify({'samples': samples, 'classCounts': class_counts})


@app.route('/available_transformations', methods=['GET'])
def get_available_transformations():
    # Return the list of available transformation names
    return jsonify(list(transformations.keys()))

#TODO: Intergrate this into react app
@app.route('/transformation_parameters', methods=['GET'])
def get_transformation_parameters():
    transformation_name = request.args.get('model')
    if transformation_name in transformations_list:
        transformation = transformations_list[transformation_name]()
        params = transformation.get_params()
        return jsonify(params)
    else:
        return jsonify({'error': 'Model not found'}), 404


@app.route('/transform', methods=['POST'])
def apply_transformation():
    data = request.json
    transformation_name = data.get('transformation')
    value = data.get('data')
    timestamp = data.get('datetime')
    # value
    print(type(value))
    value_np = np.array(value, dtype=float)

    if transformation_name in transformations:
        result = transformations[transformation_name](value_np)


        return jsonify({'result': result, 'datetime': timestamp})
    else:
        return jsonify({'error': 'Transformation not found'}), 404


@app.route('/split', methods=['POST'])
def split_data():
    data = request.json
    samples = pd.DataFrame(data['samples'])
    train_size = data.get('train_size', 0.7)
    val_size = data.get('val_size', 0.15)
    test_size = data.get('test_size', 0.15)

    if train_size + val_size + test_size != 1.0:
        return jsonify({'error': 'Train, validation, and test sizes must sum to 1.0'}), 400

    train, temp = train_test_split(samples, train_size=train_size, stratify=samples['class'])
    val, test = train_test_split(temp, test_size=test_size / (test_size + val_size), stratify=temp['class'])

    return jsonify({
        'train': train.to_dict(orient='records'),
        'val': val.to_dict(orient='records'),
        'test': test.to_dict(orient='records')
    })


@app.route('/available_models', methods=['GET'])
def get_available_models():
    # Return the list of available model names
    return jsonify(list(models.keys()))


@app.route('/model_parameters', methods=['GET'])
def get_model_parameters():
    model_name = request.args.get('model')
    if model_name in models:
        model = models[model_name]()
        params = model.get_params()
        return jsonify(params)
    else:
        return jsonify({'error': 'Model not found'}), 404


@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    model_name = data.get('model')
    parameters = data.get('parameters')
    train_data = pd.DataFrame(data.get('train_data'))
    val_data = pd.DataFrame(data.get('val_data'))

    print("Train Data: ",train_data.info())
    print("Val Data: ",val_data.info())
    print("Model Name: ",model_name)
    print("Parameters: ",parameters)

    print("Train Data: ",train_data['data'].head())

    if model_name in models:

        parameters = {key: value for key, value in parameters.items() if value is not None}
        parameters = convert_numerical_strings(parameters)

        model = models[model_name](**parameters)
        X_train, y_train = pd.DataFrame([item[0] for item in train_data['data']]), train_data['class']
        X_train.set_index('DATETIME', inplace=True)
        print("X_train: ",X_train.head())
        X_val, y_val = pd.DataFrame([item[0] for item in val_data['data']]), val_data['class']
        X_val.set_index('DATETIME', inplace=True)
        print("X_val: ", X_val.head())
        # X_val, y_val = val_data.drop(columns=['class', 'sampleIndex']), val_data['class']
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        return jsonify({'validation_score': val_score})
    else:
        return jsonify({'error': 'Model not found'}), 404


if __name__ == '__main__':
    app.run(debug=True)

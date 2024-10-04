from flask import Flask, request, jsonify, current_app
from model_service.inference import InferenceService
import config
from config import get_app_config, ModelType
from flask_sqlalchemy import SQLAlchemy
from pymodel.inference import InferenceObj, InferenceResult, db
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app_config = get_app_config()
app.config.from_object(app_config)

db_config = app_config.DB_CONFIG
app.config[
    'SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?charset={db_config['charset']}"

db.init_app(app)


@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json()
    if not data or 'input' not in data:
        return jsonify({'error': 'Invalid data'}), 400

    input_data = data['input']
    input_model = data['modelType']
    try:
        inference_service = InferenceService(input_model)
        file_names = InferenceObj.query.filter(InferenceObj.id.in_(input_data)).with_entities(
            InferenceObj.file_name).all()
        inference_service.file_names = [file_name[0] for file_name in file_names]
        inference_service.inference()
        inference_service.save_result()

        # 返回文件名和状态的列表
        result = [{'fileName': file_name, 'status': 'segmented'} for file_name in inference_service.file_names]

    except Exception as e:
        return jsonify({'error': str(e)}), 400

    print(input_data, input_model)
    return jsonify(result), 200


@app.route('/inference/<string:modeltype>', methods=['GET'])
def read_file(modeltype):
    try:
        inference_service = InferenceService(modeltype)
        file_names = InferenceObj.query.with_entities(InferenceObj.file_name).all()
        inference_service.file_names = [file_name[0] for file_name in file_names]
        inference_service.inference()
        inference_service.save_result()
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({'file names': inference_service.file_names})


@app.route('/readFile', methods=['GET'])
def read():
    data = {
        'modelType': 'MASK_RCNN_R50'
    }
    return jsonify(data)


@app.route('/')
def index():
    return "Welcome to the Flask App"


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)

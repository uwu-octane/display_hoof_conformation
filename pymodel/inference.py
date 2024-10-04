from config import ModelType
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class InferenceObj(db.Model):
    __tablename__ = 'tb_inferenceobj'
    id = db.Column(db.Integer, primary_key=True)

    file_name = db.Column(db.String, nullable=False)
    file_data = db.Column(db.LargeBinary, nullable=True)

    inference_results = db.relationship('InferenceResult', backref='inferenceObj', lazy=True)


class InferenceResult(db.Model):
    __tablenae__ = 'inference_result'
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String, nullable=False)

    inference_obj_id = db.Column(db.Integer, db.ForeignKey('tb_inferenceobj.id'), nullable=False)
    model_name = db.Column(db.String, nullable=True)

    prediction_mask = db.Column(db.LargeBinary, nullable=True)
    conformation_result = db.Column(db.LargeBinary, nullable=True)
    contour_result = db.Column(db.LargeBinary, nullable=True)
    points_sort_result = db.Column(db.LargeBinary, nullable=True)

    dorsal_hoof_wall_length = db.Column(db.Float, nullable=True)
    weight_bearing_length = db.Column(db.Float, nullable=True)
    heel_height = db.Column(db.Float, nullable=True)
    dorsal_coronary_band_height = db.Column(db.Float, nullable=True)

    dorsal_hoof_wall_angle = db.Column(db.Float, nullable=True)
    coronary_band_angle = db.Column(db.Float, nullable=True)
    heel_angle = db.Column(db.Float, nullable=True)

    inference_time = db.Column(db.Float, nullable=True)
    inference_date = db.Column(db.DateTime, nullable=False)

import io
import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer
from flask import current_app
from PIL import Image
from shapely import Polygon
import time
from config import ModelType, get_app_config, get_devconfig_instance
from detectron2.config import get_cfg

from detectron2.engine import DefaultPredictor

from model_service.conformation import process_mask, get_contours, simplify_with_hull, get_diagonals, classify_vertices, \
    fit_lines_to_subsets, get_result_intersections, get_conformation_with_keydet, draw_contours, draw_group_points, \
    plot_conformation
from pymodel.inference import InferenceResult, InferenceObj, db


class InnerResult:
    def __init__(self, conformation, inference_time, inference_date, mask_overlay_pil,
                 conformation_pil, points_sorting_result_pil, contour_pil):
        self.conformation = conformation
        self.inference_time = inference_time
        self.inference_date = inference_date
        self.mask_overlay_pil = mask_overlay_pil
        self.conformation_pil = conformation_pil
        self.points_sorting_result_pil = points_sorting_result_pil
        self.contour_pil = contour_pil


class InferenceService:
    def __init__(self, model_type_input):
        self.model_type = self.__get_model_type__(model_type_input)
        self.config = get_devconfig_instance()
        self.model = self.load_model()
        self.file_names = []
        self.file_result_dict = {}

    def __get_model_type__(self, model_type_input):
        with current_app.app_context():
            model_type = ModelType(model_type_input)
            return model_type

    def create_model_from_cfg(self, yaml, weight):
        cfg = get_cfg()
        cfg.merge_from_file(yaml)
        cfg.MODEL.WEIGHTS = weight
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

        predictor = DefaultPredictor(cfg)
        return predictor

    def load_model(self):
        model_path = self.config.get_model_path(self.model_type)
        model_yaml = self.config.get_yaml_path(self.model_type)
        if model_path is None or model_yaml is None:
            return None

        model = self.create_model_from_cfg(model_yaml, model_path)
        return model

    def inference_once(self, image_np):
        if self.model is None:
            return None

        start_time = time.time()
        inference_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        outputs = self.model(image_np)
        instance = outputs['instances']
        masks = instance.pred_masks.numpy()
        height, width = image_np.shape[:2]

        polys = []
        for mask in masks:
            mask = mask.astype(np.uint8)
            resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            processed_mask = process_mask(resized_mask)
            contours = get_contours(processed_mask)
            contours_poly = Polygon(contours[0])
            polys.append(contours_poly)

        if len(polys) == 0:
            return None

        index, max_poly = max(enumerate(polys), key=lambda x: x[1].area)

        max_mask = masks[index]

        simplified_poly = Polygon(simplify_with_hull(max_poly))

        conformation_result = get_conformation_with_keydet(max_poly, simplified_poly, weighted=True, normalized=False)
        conformation = conformation_result['conformation']
        intersections = conformation_result['intersections']
        end_time = time.time()

        inference_time = end_time - start_time
        contour_pil = draw_contours(image_np.copy(), max_poly, background=True)
        points_sorting_result_pil = draw_group_points(image_np.copy(), max_poly)

        v = Visualizer(image_np, metadata={}, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        mask_overlay = out.get_image()[:, :, ::-1]
        mask_overlay_pil = Image.fromarray(mask_overlay)

        conformation_pil = plot_conformation(intersections, conformation, image_np)

        # conformation['dorsal_hoof_wall_length']
        # conformation['weight_bearing_length']
        # conformation['heel_height']
        # conformation['dorsal_coronary_band_height']
        result = InnerResult(conformation=conformation, inference_time=inference_time,
                             inference_date=inference_date,
                             mask_overlay_pil=mask_overlay_pil, conformation_pil=conformation_pil,
                             points_sorting_result_pil=points_sorting_result_pil, contour_pil=contour_pil)
        return result

    def image_to_bytes(self, image, format='PNG'):
        # Create a BytesIO object
        byte_io = io.BytesIO()

        # Save the image to the BytesIO object in the specified format
        image.save(byte_io, format=format)

        # Get the byte data from the BytesIO object
        byte_data = byte_io.getvalue()

        # Close the BytesIO object
        byte_io.close()

        return byte_data

    def convert_bytes_to_image(self, file_data):
        image = Image.open(io.BytesIO(file_data))
        image_np = np.array(image)
        return image_np

    def inference(self):
        if self.file_names is None:
            return None

        for file_name in self.file_names:
            inference_obj = InferenceObj.query.filter_by(file_name=file_name).with_for_update(read=True).first()
            if inference_obj is None:
                continue
            file_data = inference_obj.file_data
            image_np = self.convert_bytes_to_image(file_data)
            result = self.inference_once(image_np)
            if result is not None:
                self.file_result_dict[file_name] = result

    def save_result(self):
        for key, result in self.file_result_dict.items():
            inference_obj = InferenceObj.query.filter_by(file_name=key).with_for_update(read=True).first()
            if inference_obj is None:
                continue

            inference_result = InferenceResult(file_name=key, inference_obj_id=inference_obj.id,
                                               model_name=self.model_type.name,
                                               dorsal_hoof_wall_length=result.conformation['dorsal_hoof_wall_length'],
                                               weight_bearing_length=result.conformation['weight_bearing_length'],
                                               heel_height=result.conformation['heel_height'],
                                               dorsal_coronary_band_height=result.conformation[
                                                   'dorsal_coronary_band_height'],
                                               dorsal_hoof_wall_angle=result.conformation['dorsal_hoof_wall_angle'],
                                               coronary_band_angle=result.conformation['coronary_band_angle'],
                                               heel_angle=result.conformation['heel_angle'],
                                               inference_time=result.inference_time,
                                               inference_date=result.inference_date,
                                               contour_result=self.image_to_bytes(result.contour_pil),
                                               conformation_result=self.image_to_bytes(result.conformation_pil),
                                               points_sort_result=self.image_to_bytes(
                                                   result.points_sorting_result_pil),
                                               prediction_mask=self.image_to_bytes(result.mask_overlay_pil))
            db.session.add(inference_result)
        db.session.commit()

import os
from enum import Enum


class ModelType(Enum):
    Cascade_RCNN = "Cascade_Mask_RCNN"
    MASK_RCNN_R50 = "Mask_RCNN_R50"
    MASK_RCNN_R101 = "Mask_RCNN_R101"
    ALL = "ALL"


class Config:
    DEBUG = False
    TESTING = False

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    SECRET_KEY = os.environ.get('LOG_FILE', 'app.log')


class DevelopmentConfig(Config):
    DEBUG = True
    ENV = 'development'
    CASCADE_MASK_RCNN_MODEL_PATH = '/path/to/CNNServer/models/cascade_rcnn.pth'
    MASK_RCNN_R50_MODEL_PATH = '/path/to/CNNServer/models/mr50.pth'
    MASK_RCNN_R101_MODEL_PATH = '/path/to/PycharmProjects/CNNServer/models/mr101.pth'
    CASCADE_MASK_RCNN_YAML_PATH = '/path/to/PycharmProjects/CNNServer/models/cascade_mask_rcnn_R_50_FPN_3x.yaml'
    MASK_RCNN_R50_YAML_PATH = '/path/to/PycharmProjects/CNNServer/models/mask_rcnn_R_50_FPN_3x.yaml'
    MASK_RCNN_R101_YAML_PATH = '/path/to/PycharmProjects/CNNServer/models/mask_rcnn_R_101_FPN_3x.yaml'

    DB_CONFIG = {'host': 'localhost', 'port': 0000, 'user': 'root',
                 'password': 'testpassword', 'database': 'cnndeployment',
                 'charset': 'utf8mb4', 'cursorclass': 'DictCursor'
                 }

    def get_model_path(self, model_type):
        if model_type == ModelType.Cascade_RCNN:
            return self.CASCADE_MASK_RCNN_MODEL_PATH
        elif model_type == ModelType.MASK_RCNN_R50:
            return self.MASK_RCNN_R50_MODEL_PATH
        elif model_type == ModelType.MASK_RCNN_R101:
            return self.MASK_RCNN_R101_MODEL_PATH
        else:
            return None

    def get_yaml_path(self, model_type):
        if model_type == ModelType.Cascade_RCNN:
            return self.CASCADE_MASK_RCNN_YAML_PATH
        elif model_type == ModelType.MASK_RCNN_R50:
            return self.MASK_RCNN_R50_YAML_PATH
        elif model_type == ModelType.MASK_RCNN_R101:
            return self.MASK_RCNN_R101_YAML_PATH
        else:
            return None


class TestingConfig(Config):
    TESTING = True
    ENV = 'testing'
    CASCADE_MASK_RCNN_MODEL_PATH = 'models/cascade_rcnn.pth'
    MASK_RCNN_R50_MODEL_PATH = 'models/mr50.pth'
    MASK_RCNN_R101_MODEL_PATH = 'models/mr101.pth'


class ProductionConfig(Config):
    ENV = 'production'
    CASCADE_MASK_RCNN_MODEL_PATH = 'models/cascade_rcnn.pth'
    MASK_RCNN_R50_MODEL_PATH = 'models/mr50.pth'
    MASK_RCNN_R101_MODEL_PATH = 'models/mr101.pth'


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
}


def get_app_config():
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, DevelopmentConfig())


def get_devconfig_instance():
    return DevelopmentConfig()

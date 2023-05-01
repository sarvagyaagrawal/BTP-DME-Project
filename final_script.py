import sys
import subprocess
try:
    import detectron2
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'git+https://github.com/facebookresearch/detectron2.git'])
from detectron2.utils.logger import setup_logger
setup_logger()


# implement pip as a subprocess:


# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import base64
import math

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.data.catalog import Metadata

my_metadata = Metadata()
my_metadata.set(thing_classes = ['DRT', 'CME', 'SRD'])


def get_annotations(image_base64:str):
    
    cfg1 = get_cfg()
    cfg1.merge_from_file("custommask_rcnn_R_50_FPN_3x_my_dataset.yaml")
    cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 # set threshold for this model
    cfg1.MODEL.WEIGHTS = "mymodel_0_new1.pth"
    cfg1.MODEL.DEVICE = "cpu"
    # Create predictor
    # predictor1 = DefaultPredictor(cfg)
    cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor1 = DefaultPredictor(cfg1)
    image_base64=image_base64.ljust((int)(math.ceil(len(image_base64) / 4)) * 4, '=')
    # im = cv2.imread("Copy of DME-119840-1.jpg")
    nparr = np.frombuffer(base64.b64decode(image_base64), np.uint8)
    im = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    outputs = predictor1(im)
    # print(outputs["instances"].pred_boxes)
    v = Visualizer(im[:, :, ::-1],
                   metadata=my_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    final_img=v.get_image()[:, :, ::-1]
    final_img_st = cv2.imencode('.JPEG', final_img)[1]
    base64_data = str(base64.b64encode(final_img_st))[2:-1]
    return {"annotatedImage": base64_data}

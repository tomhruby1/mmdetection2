## mmdetection v2 inference for 

from mmdet.apis import init_detector, inference_detector
from pathlib import Path
import mmcv
import sys
from tqdm import tqdm
import mmdet.datasets.coco 
from mmdet.datasets import build_dataset
from mmcv import Config
import pickle
import json
import typing as T
import numpy as np
from PIL import Image

DEBUG = False

# Specify the path to model config and checkpoint file
CONFIG_FILE = "config_v2_cascade_rcnn_r101_metacentrum_traffic_signs.py"    #'config_v2_faster_rcnn_r101_metacentrum_traffic_signs.py'
CHECKPOINT_FILE = "checkpoints/cascade_rcnn_r101_epoch_12.pth"  #'checkpoints/faster_rcnn_r101_fpn_2x_epoch_7.pth'
SENSORS = ["cam0", "cam1", "cam2", "cam3", "cam5"] # omit cam4 ... which is looking up

# mapping cat_id -> cat_name # TODO: move somewhere proper
SIGNS_CATEGORY_NAMES = (
    'A10', 'A11', 'A12', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A1a', 'A1b', 'A22', 'A24', 'A28', 'A29', 'A2a', 'A2b', 'A30', 'A31a', 'A31b', 'A31c', 'A32a', 'A32b', 'A4', 'A5a', 'A6a', 'A6b', 'A7a', 'A8', 'A9', 'B1', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B19', 'B2', 'B20a', 'B20b', 'B21a', 'B21b', 'B24a', 'B24b', 'B26', 'B28', 'B29', 'B32', 'B4', 'B5', 'B6', 'C1', 'C10a', 'C10b', 'C13a', 'C14a', 'C2a', 'C2b', 'C2c', 'C2d', 'C2e', 'C2f', 'C3a', 'C3b', 'C4a', 'C4b', 'C4c', 'C7a', 'C9a', 'C9b', 'E1', 'E11', 'E11c', 'E12', 'E13', 'E2a', 'E2b', 'E2c', 'E2d', 'E3a', 'E3b', 'E4', 'E5', 'E6', 'E7a', 'E7b', 'E8a', 'E8b', 'E8c', 'E8d', 'E8e', 'E9', 'I2', 'IJ1', 'IJ10', 'IJ11a', 'IJ11b', 'IJ14c', 'IJ15', 'IJ2', 'IJ3', 'IJ4a', 'IJ4b', 'IJ4c', 'IJ4d', 'IJ4e', 'IJ5', 'IJ6', 'IJ7', 'IJ8', 'IJ9', 'IP10a', 'IP10b', 'IP11a', 'IP11b', 'IP11c', 'IP11e', 'IP11g', 'IP12', 'IP13a', 'IP13b', 'IP13c', 'IP13d', 'IP14a', 'IP15a', 'IP15b', 'IP16', 'IP17', 'IP18a', 'IP18b', 'IP19', 'IP2', 'IP21', 'IP21a', 'IP22', 'IP25a', 'IP25b', 'IP26a', 'IP26b', 'IP27a', 'IP3', 'IP31a', 'IP4a', 'IP4b', 'IP5', 'IP6', 'IP7', 'IP8a', 'IP8b', 'IS10b', 'IS11a', 'IS11b', 'IS11c', 'IS12a', 'IS12b', 'IS12c', 'IS13', 'IS14', 'IS15a', 'IS15b', 'IS16b', 'IS16c', 'IS16d', 'IS17', 'IS18a', 'IS18b', 'IS19a', 'IS19b', 'IS19c', 'IS19d', 'IS1a', 'IS1b', 'IS1c', 'IS1d', 'IS20', 'IS21a', 'IS21b', 'IS21c', 'IS22a', 'IS22c', 'IS22d', 'IS22e', 'IS22f', 'IS23', 'IS24a', 'IS24b', 'IS24c', 'IS2a', 'IS2b', 'IS2c', 'IS2d', 'IS3a', 'IS3b', 'IS3c', 'IS3d', 'IS4a', 'IS4b', 'IS4c', 'IS4d', 'IS5', 'IS6a', 'IS6b', 'IS6c', 'IS6e', 'IS6f', 'IS6g', 'IS7a', 'IS8a', 'IS8b', 'IS9a', 'IS9b', 'IS9c', 'IS9d', 'O2', 'P1', 'P2', 'P3', 'P4', 'P6', 'P7', 'P8', 'UNKNOWN', 'X1', 'X2', 'X3', 'XXX', 'Z2', 'Z3', 'Z4a', 'Z4b', 'Z4c', 'Z4d', 'Z4e', 'Z7', 'Z9'
) 

def result2coco(result, category_names) -> list:
    ''' 
    transcribe mmdet2 detector bbox result into coco formatted dictionary
    :param: result: mmdet2 result object 
    :param: label: frame of detections description e.g.: cam0, image_id...
    :result: list of coco style detection dicts
    '''
    predictions = []
    for cat_idx, cat_res in enumerate(result):
        if cat_res.size != 0:
            for pred in cat_res: # over per-cat predictions
                pred = [float(p) for p in pred]  # retype to std float for serialization
                # bboxes to coco
                coco_bbox = [-1]*4
                coco_bbox[:2] = pred[:2]
                coco_bbox[2] = pred[2] - pred[0]
                coco_bbox[3] = pred[3] - pred[1]
                
                predictions.append({
                    "bbox": coco_bbox, 
                    "score": pred[4],
                    "category_id": cat_idx, 
                    "category_name": category_names[cat_idx],
                    # "area":
                })   
    return predictions

class ReelInference:
    DETECTION_OUTPUT_FILE = "detections.json"
    def __init__(self, config_file, checkpoint_file, category_names: list, output_path, device="cuda:0"):
        self.category_names = category_names
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.cfg = Config.fromfile(config_file) # mmdetection config file
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        if DEBUG:
            self.debug_dir = self.output_path/'debug'
            self.debug_dir.mkdir(exist_ok=True)

    def store_detections(self, detections: dict, indent=None):
        with open(self.output_path/ReelInference.DETECTION_OUTPUT_FILE, 'w') as f:
            json.dump(detections, f, indent=indent)

    def reel_inference(self, data_p: T.Union[str, Path], copy_and_rot=True):
        '''
        Running inference on extracted reel
        :param: data_p path to datadir expected to have cam0, cam1, ...,cam5 subdirs
        '''            
        detections = {}
        for sensor in SENSORS:
            cam_frame_ps = sorted(list((Path(data_p)/sensor).glob('*.jpg'))) # get camera frame img paths
            if len(cam_frame_ps) < 1:
                print(f"No camera frames found for cam {sensor}. Quitting.")
                return
            print(f"processing {sensor}...")
            for p in tqdm(cam_frame_ps):
                img = mmcv.imread(str(p))
                img = np.rot90(img) # TODO: obtain rotation from camtype
                result = inference_detector(self.model, img)
                detections[p.name] = result2coco(result, self.category_names)
                self.store_detections(detections) # TODO: how often write to a disk?
                if DEBUG:
                    out_file = self.debug_dir/p.name
                    self.model.show_result(img, result, out_file=str(out_file))
            self.store_detections(detections, indent=2)

# docker run -it -v ./:/root/mmdet2_mount -v /media/tomas/samQVO_4TB_D/:/data --gpus all mmdet2
if __name__=='__main__':
    if len(sys.argv) != 3:
        data_dir = "/data/asset-detection-datasets/drtinova_med_u_track/data_m/reel_undistorted" #/data/reel_0003_20221010-124708/extracted_all/data/reel_0003_20221010-124708/"
        out_p = "/data/asset-detection-datasets/drtinova_med_u_track/data_m/detections"
    else:
        data_dir = sys.argv[1]
        out_p = sys.argv[2]

    ri = ReelInference(CONFIG_FILE, CHECKPOINT_FILE, SIGNS_CATEGORY_NAMES, out_p)
    ri.reel_inference(data_dir)
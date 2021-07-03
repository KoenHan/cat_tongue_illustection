import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
from copy import deepcopy
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from IPython.display import display
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

import mask_detect
from headpose import est_face_pose
from utils import Singleton

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

'''
masking modelをシングルトン化
普通のコンストラクタによる呼び出しを禁止したいけど，
pythonだと頑張れば回避できちゃうので，
コンストラクタ使うのやめてね
'''
class MaskingModel(Singleton):
    def __init__(self):
        model_name = "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
        base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_dir = tf.keras.utils.get_file(
                                    fname=model_name,
                                    origin=base_url + model_name + '.tar.gz',
                                    untar=True)
        model_dir = pathlib.Path(model_dir)/"saved_model"
        self.model = tf.saved_model.load(str(model_dir))

'''
category indexをシングルトン化
普通のコンストラクタによる呼び出しを禁止したいけど，
pythonだと頑張れば回避できちゃうので，
コンストラクタ使うのやめてね
'''
class CategoryIndex(Singleton):
    def __init__(self):
        # Loading label map
        PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        self.category_index

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            # output_dict['detection_masks'], # これだとなぜか途中で躓いて実行できないので以下に変える
            tf.convert_to_tensor(output_dict['detection_masks']), # 参考：https://blog.csdn.net/weixin_33423187/article/details/115331115
            output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def show_inference(image_np):
    model = MaskingModel.get_instance().model
    output_dict = run_inference_for_single_image(model, image_np)
    #検出した矩形ごとにマスク検出、上半身検出に入れる。
    output_dict['mask_flag'] = [False for i in range(output_dict['num_detections'])]
    output_dict['face_box'] = [None for i in range(output_dict['num_detections'])]
    output_dict['upper_flag'] = [False for i in range(output_dict['num_detections'])]
    output_dict['face_direction'] = [0 for i in range(output_dict['num_detections'])]
    HEAD_UPPER_ASPECT = 4 # 頭の長さ/上半身の長さ．とりあえず7頭身を想定
    LOW_SCORE = 0.5 # 検出確率の閾値

    for i in range(output_dict['num_detections']):
        (h, w) = image_np.shape[:2]
        output_dict['detection_boxes'][i]
        b_box = output_dict['detection_boxes'][i]
        (left, right, top, bottom) = deepcopy((int(b_box[1] * w), int(b_box[3] * w), int(b_box[0] * h), int(b_box[2] * h)))
        output_dict['detection_boxes'][i] = np.array([top, left, bottom, right], dtype=np.int64)

        if output_dict['detection_classes'][i] == 1 :
            # マスク検出
            person_image = image_np[top:bottom, left:right]

            if output_dict['detection_scores'][i] < LOW_SCORE :
                # 可能性が低いものはマスク検出と上半身判定を飛ばす
                continue
            mask_res = mask_detect.mask_image(person_image)
            if mask_res is None :
                # マスク検出は顔検出を兼ねているので，
                # 検出できなかった場合は顔も検出されてないので，上半身の検出もスキップ
                continue
            fbox, output_dict['mask_flag'][i] = mask_res
            # 顔の領域を絶対座標に変換
            # tfのtop, left, bottom, right順に揃える
            output_dict['face_box'][i] = np.array([
                top + fbox[1], left + fbox[0], top + fbox[3], left + fbox[2]
            ], dtype=np.int64)

            # 上半身判定
            output_dict['upper_flag'][i] = \
                (bottom - top)/(output_dict['face_box'][i][2] - output_dict['face_box'][i][0]) <= HEAD_UPPER_ASPECT

            # 顔の向き判定
            _, _, yaw, _ = est_face_pose(image_np, output_dict['face_box'][i]) # 顔の姿勢の推測，向きがわかればいいのでyawのみ取っている
            output_dict['face_direction'][i] = int(yaw) # 正なら左向き，負なら右向き，0なら正面
    output_dict['detection_boxes'] = output_dict['detection_boxes'].astype(np.int64)
    return output_dict

if __name__ == "__main__" :
    def output_dict_test():
        # set test images path
        # PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
        PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images/')
        TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
        for image_path in TEST_IMAGE_PATHS:
            img = cv2.imread(str(image_path))
            res = show_inference(img)
            print(image_path)
            print(res)

            for idx, detection_class in enumerate(res['detection_classes']) :
                dbox = res['detection_boxes'][idx]
                fbox = res['face_box'][idx]
                if dbox is not None and fbox is not None :
                    # (left, top), (right, bottom)
                    cv2.rectangle(img, (fbox[1], fbox[0]), (fbox[3], fbox[2]), (0, 255, 0), 10)
                    cv2.rectangle(img, (dbox[1], dbox[0]), (dbox[3], dbox[2]), (0, 255, 255) if res['upper_flag'][idx] else (0, 0, 255), 10)
                elif dbox is not None and res['detection_scores'][idx] >= 0.5: # 0.5はshow_infereceのLOW_SCOREの値を写してきただけ
                    # 顔がないので上半身のみかどうかの判定のしようがない
                    cv2.rectangle(img, (dbox[1], dbox[0]), (dbox[3], dbox[2]), (0, 0, 255), 10)
            img = cv2.resize(img, (1600, 900))
            img_name = str(image_path).split('/')[-1]
            cv2.imwrite('images/face_mask_detect_res/'+img_name, img)

    def head_pose_est_test():
        PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images/')
        TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
        for image_path in TEST_IMAGE_PATHS:
            print(image_path)
            img = cv2.imread(str(image_path))
            res = show_inference(deepcopy(img))

            for idx, detection_class in enumerate(res['detection_classes']) :
                dbox = res['detection_boxes'][idx]
                fbox = res['face_box'][idx]
                if dbox is not None and fbox is not None :
                    _, _, yaw, masks = est_face_pose(img, fbox) # 顔の姿勢の推定
                    for p in masks:
                        cv2.drawMarker(img, (int(p[0]), int(p[1])), (255, 0, 255), thickness=10)
                    fcenter = ((fbox[1] + fbox[3])//2, (fbox[0] + fbox[2])//2)
                    if yaw < -1.0 :
                        arw_dst_j = fbox[3]
                    elif yaw > 1.0 :
                        arw_dst_j = fbox[1]
                    else :
                        arw_dst_j = fcenter[0]
                    arw_dst_j = fbox[3] if yaw < 0 else fbox[1]
                    (left, top), (right, bottom)
                    cv2.rectangle(img, (fbox[1], fbox[0]), (fbox[3], fbox[2]), (0, 255, 0), 10)
                    cv2.rectangle(img, (dbox[1], dbox[0]), (dbox[3], dbox[2]), (0, 255, 255) if res['upper_flag'][idx] else (0, 0, 255), 10)
                    cv2.arrowedLine(img, fcenter, (arw_dst_j, fcenter[1]), (255, 0, 0), 10)
                elif dbox is not None and res['detection_scores'][idx] >= 0.5: # 0.5はshow_infereceのLOW_SCOREの値を写してきただけ
                    # 顔がないので上半身のみかどうかの判定のしようがない
                    cv2.rectangle(img, (dbox[1], dbox[0]), (dbox[3], dbox[2]), (0, 0, 255), 10)
            # img = cv2.resize(img, (1600, 900))
            # img_name = str(image_path).split('/')[-1]
            # cv2.imwrite('images/head_pose_est/'+img_name, img)

    # output_dict_test()
    head_pose_est_test()

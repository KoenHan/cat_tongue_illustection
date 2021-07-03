from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
from PIL import Image
import pathlib

from utils import Singleton

'''
masking modelをシングルトン化
普通のコンストラクタによる呼び出しを禁止したいけど，
pythonだと頑張れば回避できちゃうので，
コンストラクタ使うのやめてね
'''
class MaskDetectModel(Singleton):
    def __init__(self):
        model = "./mask_detect/mask_detector.model"
        face = "./mask_detect/face_detector"

        # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        print("[INFO] loading face mask detector model...")
        prototxtPath = os.path.sep.join([face, "deploy.prototxt"])
        weightsPath = os.path.sep.join([face, "res10_300x300_ssd_iter_140000.caffemodel"])

        # load the face mask detector model from disk
        self.model = {
            'face_detect' : cv2.dnn.readNet(prototxtPath, weightsPath),
            'face_mask_detect' : load_model(model)}

def mask_image(image, confidence=0.5):
    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    # image = cv2.imread(image)
    model = MaskDetectModel.get_instance().model
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    model['face_detect'].setInput(blob)
    detections = model['face_detect'].forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        conf = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if conf > confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            if face.size == 0 :
                # faceが空になるとき，cv2.cvtColorで実行エラーになるのでスキップ
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model['face_mask_detect'].predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            if(label == "Mask"):
                return box.astype("int"), True
            else:
                return box.astype("int"), False


if __name__ == "__main__":
    PATH_TO_TEST_IMAGES_DIR = pathlib.Path('models/research/object_detection/test_images')
    TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
    for image_path in TEST_IMAGE_PATHS :
        image = cv2.imread(str(image_path))
        print(smask_image(image))
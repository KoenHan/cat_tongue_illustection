import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
from headpose import DIR
import time
import os

from utils import Singleton

class PoseEstimator(Singleton):
    def __init__(self):
        self.model = {
            'pose_est' : keras.models.load_model(DIR/"model"/"pose_model"),
            'model_points' : np.loadtxt(DIR/"model"/"model_points.txt")}

def detect_marks(pose_est_model, image_np):
    """Detect marks from image"""
    # Actual detection.
    predictions = pose_est_model.signatures["predict"](
        tf.constant(image_np, dtype=tf.uint8))
    # Convert predictions to landmarks.
    marks = np.array(predictions['output']).flatten()[:136]
    marks = np.reshape(marks, (-1, 2))
    return marks

def detect_face_marks(pose_est_model, image, face_box):
    '''
    渡された顔の画像に対して特徴量検出を行い，検出された特徴点を返す

    Params:
        pose_est_model: (略)
            顔の姿勢推定用モデル
        image: ndarray
            入力画像
        face_box: List
            顔を囲っているバウンディングボックス
            [top, left, bottom, right]形式

    Returns:
        marks: ndarray
            顔の特徴点の座標の配列
            marks.shape = (特徴点の数，2)
            marks[*] = [行方向の座標，列方向の座標]
    '''
    face_img = image[face_box[0]:face_box[2], face_box[1]:face_box[3]]
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    marks = detect_marks(pose_est_model, [face_img])
    marks[:, 0] *= (face_box[3] - face_box[1])
    marks[:, 0] += face_box[1]
    marks[:, 1] *= (face_box[2] - face_box[0])
    marks[:, 1] += face_box[0]
    return marks

def est_face_pose(image, face_box):
    '''
    渡された顔の画像に対して，顔の姿勢を推定

    Params:
        image: ndarray
            入力画像
        face_box: List
            顔を囲っているバウンディングボックス
            [top, left, bottom, right]形式

    Returns:
        -: Tuple
            顔の姿勢と特徴点
            姿勢は[roll, pitch, yaw]形式
            特徴点は検出結果の画像を表示するためだけに返している
    '''
    model = PoseEstimator.get_instance().model

    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype="double")

    extend_i = (face_box[2] - face_box[0])//10
    extend_j = (face_box[3] - face_box[1])//10
    top = face_box[0] - extend_i
    left = face_box[1] - extend_j
    bottom = face_box[2] + extend_i
    right = face_box[3] + extend_j
    # 若干大きくした方が結果が良くなった気がする
    facebox = [
        top if top >= 0 else face_box[0],
        left if left >= 0 else face_box[1],
        bottom if bottom <= image.shape[0] else image.shape[0],
        right if right <= image.shape[1] else image.shape[1]]
    marks = detect_face_marks(model['pose_est'], image, facebox)
    shape = marks.astype(np.uint)
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26],
                            shape[36], shape[39], shape[42], shape[45],
                            shape[31], shape[35], shape[48], shape[54],
                            shape[57], shape[8]])
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (_, rotation_vec, translation_vec) = \
        cv2.solvePnP(model['model_points'], image_pts, camera_matrix, dist_coeffs)

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
    angles[0, 0] = angles[0, 0] * -1

    return angles[1, 0], angles[0, 0], angles[2, 0], marks  # roll, pitch, yaw, 特徴点

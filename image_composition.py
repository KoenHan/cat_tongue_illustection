import cv2
import numpy as np
import itertools
import time

from functools import cmp_to_key
from itertools import combinations

import utils

def blur_image_np(image, masks, blur_kernel_size=0.000009, dilate_kernel_size=0.000008):
    '''
    検出結果を元に該当部分にブラーをかける

    Params:
        image: ndarray
            画像のndarray
        masks: ndarray
            マスクのndarray
        blur_kernel_size
            ブラーをかける際に用いるカーネル関数の大きさ(大きくするほど強くぼやける)
        dilate_kernel_size
            モルフォロジー処理（膨張）をする際に用いるカーネル関数の大きさ(大きくするほどマスクが大きく膨張する)

    Returns:
        ndarray
            ブラー処理を施したndarray
    '''
    height, width = image.shape[:2]
    blur_kernel_size = int(blur_kernel_size * height * width)
    dilate_kernel_size = int(dilate_kernel_size * height * width)

    # 全てのマスクを１枚のマスクに
    if len(masks) == 0:
        return image
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)
    mask = np.clip(mask * 255, 0, 255)

    # 膨張（モルフォロジー演算）
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, dilate_kernel)

    # 元画像全体にブラーをかける
    blurred_image = cv2.blur(image, (blur_kernel_size, blur_kernel_size))

    # マスクがかかっていない範囲は元に戻す
    blurred_image[dilated_mask == 0] = image[dilated_mask == 0]

    return blurred_image


# 2つの矩形の共通面積を返す
def common_area(rec1, rec2):
    top1, left1, bottom1, right1 = rec1
    top2, left2, bottom2, right2 = rec2

    # rec1 の左上 (top1, left1) が rec2 の中
    if left2 < left1 and left1 < right2 and top2 < top1 and top1 < bottom2:
        return (right2 - left1) * (bottom2 - top1)
    # rec1 の右上 (top1, right1) が rec2 の中
    if left2 < right1 and right1 < right2 and top2 < top1 and top1 < bottom2:
        return (right1 - left2) * (bottom2 - top1)
    # rec1 の左下 (bottom1, left1) が rec2 の中
    if left2 < left1 and left1 < right2 and top2 < bottom1 and bottom1 < bottom2:
        return (right2 - left1) * (bottom1 - top2)
    # rec1 の右下 (bottom1, right1) が rec2 の中
    if left2 < right1 and right1 < right2 and top2 < bottom1 and bottom1 < bottom2:
        return (right1 - left2) * (bottom1 - top2)

    return 0


# 2つの矩形の距離を返す
def box_dist(rec1, rec2):
    top1, left1, bottom1, right1 = rec1
    top2, left2, bottom2, right2 = rec2

    # rec1 の右に rec2 がある
    if right1 < left2:
        if (top1 <= top2 and top2 <= bottom1) or (top1 <= bottom2 and bottom2 <= bottom1):
            return left2 - right1
        elif bottom2 < top1:
            return np.sqrt((left2-right1)*(left2-right1)+(top1-bottom2)*(top1-bottom2))
        elif bottom1 < top2:
            return np.sqrt((left2-right1)*(left2-right1)+(bottom1-top2)*(bottom1-top2))
    # rec1 の左に rec2 がある
    elif right2 < left1:
        if (top1 <= top2 and top2 <= bottom1) or (top1 <= bottom2 and bottom2 <= bottom1):
            return left1 - right2
        elif bottom2 < top1:
            return np.sqrt((left1-right2)*(left1-right2)+(top1-bottom2)*(top1-bottom2))
        elif bottom1 < top2:
            return np.sqrt((left1-right2)*(left1-right2)+(bottom1-top2)*(bottom1-top2))
    else:
        if (top1 <= top2 and top2 <= bottom1) or (top1 <= bottom2 and bottom2 <= bottom1):
            return 0
        elif bottom2 < top1:
            return top1 - bottom2
        elif bottom1 < top2:
            return top2 - bottom1
    
    return 0


def merge_object(detected_results):
    '''
    物体検知の結果を元に近い物体を統合する

    Params:
        detected_results: List
            ((矩形),クラスid) のリスト
    Returns:
        List
            統合した結果となる ([(矩形1),…],[クラスid,…]) のリスト
    '''
    # 距離100以下の矩形ペアを統合候補に追加
    merge_dist = 100
    merge_candidate = []
    for idx1, idx2 in itertools.combinations(range(len(detected_results)), 2):
        box1, box2 = detected_results[idx1][0], detected_results[idx2][0]
        dist = box_dist(box1, box2)
        if dist < merge_dist:
            carea = 0
            if dist == 0:
                carea = common_area(box1, box2)
            merge_candidate.append((-carea, dist, idx1, idx2))
    merge_candidate = sorted(merge_candidate)

    # 共通する領域が大きいものから統合
    delete_flag = [False for _ in range(len(detected_results))]
    merged_result = []
    for _, _, idx1, idx2 in merge_candidate:
        if delete_flag[idx1] or delete_flag[idx2]:
            continue
        obj1, obj2 = detected_results[idx1], detected_results[idx2]
        if obj1[1] == obj2[1] or len(utils.get_illust_path([obj1[1], obj2[1]])) == 0:
            continue
        delete_flag[idx1], delete_flag[idx2] = True, True
        merged_result.append(([obj1[0], obj2[0]], [obj1[1], obj2[1]]))

    # 統合されなかったものを追加
    for i in range(len(detected_results)):
        if delete_flag[i] == False:
            res = detected_results[i]
            merged_result.append(([res[0]], [res[1]]))

    return merged_result


# 前後関係の推定で面積順でソートするための比較関数
def cmp_area(l, r):
    size = {1: 3.8, 2: 2.9, 3: 1, -4: 7.6, -5: 7.6, -6: 7.6, -7: 3.8, -8: 3.8, -9: 3.8, 10: 25, 27: 12.5, 65: 2.6, 73: 16.6}
    l_area, r_area = 0, 0
    for top, left, bottom, right in l[0]:
        l_area += (bottom - top) * (right - left)
    for top, left, bottom, right in r[0]:
        r_area += (bottom - top) * (right - left)

    if l[1][0] in size:
        l_area *= size[l[1][0]]
    if r[1][0] in size:
        r_area *= size[r[1][0]]

    if l_area == r_area:
        return 0
    if l_area < r_area:
        return -1
    return 1

def resize_illust(box, image_size, illust, magnification_rate):
    top, left, bottom, right = box
    height, width = image_size

    # リサイズ
    illust_height, illust_width = illust.shape[:2]
    ratio = min((bottom - top) / illust_height, (right - left) / illust_width) + magnification_rate
    illust_rs = cv2.resize(illust, dsize=None, fx=ratio, fy=ratio)
    # 貼り付ける位置を計算
    box_center_x, box_center_y = (left + right) // 2, (top + bottom) // 2
    illust_height, illust_width = illust_rs.shape[:2]
    paste_top = box_center_y - (illust_height // 2)
    paste_bottom = paste_top + illust_height
    paste_left = box_center_x - (illust_width // 2)
    paste_right = paste_left + illust_width

    # 拡大した結果、写真を飛び出す場合は拡大しない
    if paste_top < 0 or paste_bottom >= height or paste_left < 0 or paste_right >= width:
        # リサイズ
        illust_height, illust_width = illust.shape[:2]
        ratio = min((bottom - top) / illust_height, (right - left) / illust_width)
        illust_rs = cv2.resize(illust, dsize=None, fx=ratio, fy=ratio)
        # 貼り付ける位置を計算
        box_center_x, box_center_y = (left + right) // 2, (top + bottom) // 2
        illust_height, illust_width = illust_rs.shape[:2]
        paste_top = box_center_y - (illust_height // 2)
        paste_bottom = paste_top + illust_height
        paste_left = box_center_x - (illust_width // 2)
        paste_right = paste_left + illust_width
    
    return (illust_rs, (paste_top, paste_left, paste_bottom, paste_right))

def paste_illust(image, detected_result, magnification_rate = 0.2):
    '''
    画像と物体検知の結果を元にリサイズを行ったいらすとやを貼り付け

    Params:
        image: ndarray
            入力画像
        detected_result: List
            ([(矩形1),…],[クラスid,…]) のリスト
    Returns:
        ndarray
            貼り付けた画像
    '''
    # 矩形の大きさが小さい順にsort
    detected_result = sorted(detected_result, key=cmp_to_key(cmp_area))

    height, width = image.shape[:2]
    for boxes, class_ids in detected_result:
        # 検知した矩形領域
        top, left, bottom, right = boxes[0]
        for t, l, b, r in boxes:
            top = min(top, t)
            bottom = max(bottom, b)
            left = min(left, l)
            right = max(right, r)
        # クラス名から画像を選択
        illust_paths = utils.get_illust_path(class_ids)
        if len(illust_paths) == 0:
            print("warning: empty " + str(class_ids))
            continue
        origin_illust = cv2.imread(str(illust_paths[0]), cv2.IMREAD_UNCHANGED)

        # 通常のイラスト
        illust, paste_pos = resize_illust((top, left, bottom, right), (height, width), origin_illust, magnification_rate)
        paste_top, paste_left, paste_bottom, paste_right = paste_pos
        illust_alpha = illust[:, :, 3:] / 255

        # 透過して貼り付け
        image[paste_top: paste_bottom, paste_left: paste_right] = image[paste_top: paste_bottom, paste_left: paste_right] * (1 - illust_alpha) + illust[:, :,:3] * illust_alpha

    return image


def composite_image(image, inference_result, config):
    '''
    画像と物体検出結果を元にぼかし・物体の統合・いらすとやの合成を行う

    Params:
        image: ndarray
            入力画像
        inference_result: dict
            detect_objects の出力
        config: dict
            諸設定
    Returns:
        ndarray
            合成後の画像
    '''
    # 低スコアを弾く
    detected_targets = [1, 2, 3, 27, 65, 73]
    scores = inference_result['detection_scores']
    upper_flags = inference_result['upper_flag']
    mask_flags = inference_result['mask_flag']
    face_dir = inference_result['face_direction']
    boxes0 = inference_result['detection_boxes']
    classes0 = inference_result['detection_classes']
    masks0 = inference_result['detection_masks_reframed']
    boxes, classes, masks = [], [], []
    m_boxes, m_classes = [], []
    for sc, box, cl_id, mask, uf, mf, dir in zip(scores, boxes0, classes0, masks0, upper_flags, mask_flags, face_dir):
        if sc > 0.8 and cl_id in detected_targets:
            # upper_left: -4, upper_front: -5, upper_right: -6
            # left: -7, front: -8, right: -9
            if cl_id == 1:
                if uf == True and dir > 0:
                    cl_id = -4       # upper_left
                elif uf == True and dir == 0:
                    cl_id = -5       # upper_front
                elif uf == True and dir < 0:
                    cl_id = -6       # upper_right
                elif uf == False and dir > 0:
                    cl_id = -7       # left
                elif uf == False and dir == 0:
                    cl_id = -8       # front
                elif uf == False and dir < 0:
                    cl_id = -9       # right
                else:
                    print("dir:" + str(dir))

            if mf == False:
                boxes.append(box)   # top, left, bottom, right
                masks.append(mask)
                classes.append(cl_id)
            else:
                m_boxes.append(box)
                m_classes.append(cl_id)
                masks.append(mask)

    # 検出結果をまとめる
    detected_result = []
    for i in range(len(boxes)):
        detected_result.append((boxes[i], classes[i]))
    merged_results = merge_object(detected_result)
    # マスクを追加
    for box, cl_id in zip(m_boxes, m_classes):
        merged_results.append(([box], [cl_id, 10])) # マスクのidは10

    # 画像をぼかす
    image = blur_image_np(image, masks)
    # イラストを貼る
    image = paste_illust(image, merged_results)

    return image


# 使用例
if __name__ == "__main__":
    def test():
        import obj_detect

        for image_id in range(1, 26):
            start = time.time()
            print("image_id:" + str(image_id))
            image_path = 'images/' + str(image_id) + '.jpg'
            img = cv2.imread(str(image_path))
            output_dict = obj_detect.show_inference(img)

            # いらすとやの合成
            img = cv2.imread(str(image_path))
            config = dict()
            img = composite_image(img, output_dict, config)
            # 保存
            cv2.imwrite("res" + str(image_id) + ".jpg", img)
            print("E:{}(s)".format(time.time() - start))


    test()
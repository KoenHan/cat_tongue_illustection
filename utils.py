import os
from typing import List

class Singleton(object):
    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

def get_illust_path(classes: List[int]) -> List[str]:
    '''
    与えられたクラスを全て含んでいる画像のパスを返す関数

    Params:
        classes: List[int]
            画像に含んでほしい物体のクラスのリスト
    Returns:
        List[int]
            classesを全部含んでいる画像のパス
            一枚もない場合は空のリストを返す
    '''
    # 人：1, 自転車：2, 車：3, マスク：5, 上半身：-4, バックパック：27, ベッド：65, PC（ラップトップ）：73
    id_name = {
        1 : 'front_person',
        2 : 'bicycle',
        3 : 'car',
        -4 : 'upper_left_person',
        -5 : 'upper_front_person',
        -6 : 'upper_right_person',
        -7 : 'left_person',
        -8 : 'front_person',
        -9 : 'right_person',
        10 : 'mask', 
        27 : 'backpack',
        65 : 'bed',
        73 : 'laptop'
    }
    folder_path = './illust/'
    for c in sorted(classes) :
        if c in id_name:
            folder_path += id_name[c] + '/'

    illust_path = []
    if os.path.exists(folder_path) :
        illust_path = filter(
            lambda f : os.path.isfile(os.path.join(folder_path, f)),
            os.listdir(folder_path))
        illust_path = map(lambda f : folder_path+f, illust_path)
    return list(illust_path)

if __name__ == "__main__" :
    def gen_illust_path_test():
        print(get_illust_path([1]))
        print(get_illust_path([2]))
        print(get_illust_path([1, 3]))
        print(get_illust_path([1, 27]))
        print(get_illust_path([1, 2, 3]))

    gen_illust_path_test()


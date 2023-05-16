# Автор: Кожекин Михаил Викторович
# Описание: парсим xml файлы, форматируем, делаем визуализацию и сохраняем в папку pics
# Источник данных: https://figshare.com/articles/dataset/Wheat_and_Car_Dataset_with_Rotated_Bounding_Box_Annotations/13014230
import cv2
import glob
import os
import re
import numpy as np
from math import pi, sin, cos, radians, sqrt


def rotate(point, angle_deg):
    ''' вращаем точку вокруг начала координат
        - point -- точка [lat, long] (y, x)
        - angle_deg -- угол\n
        returns:
        - Y, X  -- новая точка '''
    angle_rad = radians(angle_deg)
    sin_a = sin(angle_rad)
    cos_a = cos(angle_rad)
    y, x = point
    X = x*cos_a - y*sin_a
    Y = x*sin_a + y*cos_a
    return Y, X


def handle_rotated_xml_file(
    path_rotated_xml_file: str,
):
    ''' читаем, парсим, форматируем массив отрезков
    - path_rotated_xml_file -- путь до xml файла\n
    returns:
    - shortcuts -- массив отрезков в формате [[x1,y1], [x2,y2]], где xy1=top-left, xy2=bottom-right. shape: (n, 2, 2)'''
    # 1. парсим cx,cy,w,h,angle
    pattern = '<cx>(\w+\.\w+)<\/cx>\$?\s*<cy>(\w+\.\w+)<\/cy>\$?\s*<w>(\w+\.\w+)<\/w>\$?\s*<h>(\w+\.\w+)<\/h>\$?\s*<angle>(\w+\.\w+)<\/angle>'
    l = []
    f = open(path_rotated_xml_file, 'r')
    s = f.read()
    for i in re.findall(pattern, s, re.MULTILINE):
        l += [np.array(i).astype(float)]
    # 2. преобразуем [cx,cy,w,h,a]  =>  [p1,p2]
    shortcuts = []
    for elem in l:
        cx,cy,w,h,angle = elem
        # https://github.com/open-mmlab/mmrotate/blob/main/docs/en/intro.md#what-is-rotated-box
        # theta = angle * pi / 180 => angle = 180*theta/pi
        angle_deg = 180*angle/pi
        p  = np.array([cx, cy]).astype(int)
        len = sqrt(h**2 + w**2)/2
        v0 = np.int0([len, 0])
        v  = np.int0(rotate(v0, -angle_deg))
        shortcuts += [[p-v, p+v]]
    return np.int32(shortcuts)


def handle_axis_aligned_xml_file(
    path_axis_aligned_xml_file: str,
):
    ''' читаем, парсим, форматируем массив описанных прямоугольников
    - path_axis_aligned_xml_file -- путь до xml файла\n
    returns:
    - bboxes -- массив описанных прямоугольников в формате [[x1,y1], [x2,y2]], где xy1=top-left, xy2=bottom-right. shape: (n, 2, 2)'''
    # 1. парсим xmin, ymin, xmax, ymax
    pattern = '<xmin>(\w+)<\/xmin>$\s+<ymin>(\w+)<\/ymin>$\s+<xmax>(\w+)<\/xmax>$\s+<ymax>(\w+)<\/ymax>'
    f = open(path_axis_aligned_xml_file, 'r')
    s = f.read()
    l = []
    for i in re.findall(pattern, s, re.MULTILINE):
        l += [np.int0(i)]
    # 2. преобразуем xmin, ymin, xmax, ymax  =>  [p1,p2]
    bboxes = []
    for elem in l:
        xmin, ymin, xmax, ymax = elem
        bboxes += [[
            [xmin, ymin],
            [xmax, ymax]
        ]]
    return np.int32(bboxes)



root = '/home/user/icg_data/rotated_bbox_gwhd_2020'
folds = [
    'ethz_1',
    'inrae_1',
    'rres_1',
    'usask_1',
]
for fold in folds:
    paths_images            = sorted(glob.glob(f'{root}/{fold}/images/*'))
    paths_rotated_xml       = sorted(glob.glob(f'{root}/{fold}/rotated_xml/*.xml'))
    paths_axis_aligned_xml  = sorted(glob.glob(f'{root}/{fold}/axis_aligned_xml/*.xml'))
    print('\n'*2)
    print(fold)
    shortcuts = list(map(handle_rotated_xml_file, paths_rotated_xml))
    bboxes    = list(map(handle_axis_aligned_xml_file, paths_axis_aligned_xml))
    # визуализация
    k = 55
    img = cv2.imread(paths_images[k])
    print(paths_images[k], paths_rotated_xml[k])
    for p1, p2 in shortcuts[k]:
        img = cv2.circle(img, p1, 10, (0,255,0), 1, cv2.LINE_AA)
        img = cv2.line(img, p1, p2, (0,255,0), 1, cv2.LINE_AA)
    for p1, p2 in bboxes[k]:
        img = cv2.rectangle(img, p1, p2, (0,128,0), 1, cv2.LINE_AA)
    cv2.imwrite(f'pics/{fold}_{k}.png', img)

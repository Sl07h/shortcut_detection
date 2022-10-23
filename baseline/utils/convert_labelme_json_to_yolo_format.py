import enum
import cv2
import os
import glob
import json
import numpy as np


def read_labelme_file_to_numpy(path_markup: str) -> None:
    ''' @brief преобразует файл разметки в массив координат
    @param path_markup путь до файла разметки
    @return coordinates [[p0_x, p0_y, p1_x, p1_y], ...]
    '''
    with open(path_markup, 'r') as f:
        data = json.load(f)['shapes']
        coordinates = np.ndarray((len(data), 4), np.int64)
        for i, elem in enumerate(data):
            bbox = np.array(elem['points']).flatten()
            coordinates[i] = bbox
    return coordinates


def convert_shortcut_to_bbox(path_image: str, coordinates: np.ndarray, width_px = 20, do_draw = False) -> None:
    ''' @brief преобразует отрезок в описанный прямоугольник и сохраняет его в папку tmp/markup_check
    @param path_image путь до изображения
    @param coordinates [[p0_x, p0_y, p1_x, p1_y], ...] координаты отрезков
    @param width_px ширина объекта в пикселях
    @param do_draw = False нужна ли отрисовка координат на изображении
    @return coordinates [[p0_x, p0_y, p1_x, p1_y], ...] координаты прямоугольников

    '''
    img = cv2.imread(path_image)
    image_height_px, image_width_px, _ = img.shape

    bbox_yolo = np.ndarray((coordinates.shape[0], 5))
    for i, coords in enumerate(coordinates):
        # точки отрезка
        p1 = coords[:2].copy()
        p2 = coords[2:].copy()

        # рассчитываем вектор нормальный к главной оси
        v = p2 - p1
        v = v / np.linalg.norm(v)
        v = v[::-1]
        v[0] *= -1.0
        v = v * width_px / 2.0
        v = np.array(v, np.int64)

        # считаем новый bbox
        points = np.array([
            p1 + v,
            p1 - v,
            p2 + v,
            p2 - v,
        ]).T

        # ограничиваем bbox размерами изображения
        def replace_with_height(l):
            l[l>image_height_px] = image_height_px - 1
            return l
        points[points < 0] = 0
        points[points > image_width_px]  = image_width_px
        points[1] = replace_with_height(points[1])
        p1_new = np.array([
            points[0].min(),
            points[1].min(),
        ])
        p2_new = np.array([
            points[0].max(),
            points[1].max(),
        ])
        # 0 if binary classification https://github.com/AlexeyAB/Yolo_mark/issues/60
        bbox_yolo[i] = np.array([0, *p1_new, *p2_new])  

        if do_draw:
            cv2.rectangle(img, p1_new, p2_new, (0,255,0), 1)
            img = cv2.line(img, p1, p2, (255,0,0), 1, cv2.LINE_AA)
            img = cv2.line(img, p1 + v, p1 - v, (255,0,0), 1, cv2.LINE_AA)
            img = cv2.line(img, p2 + v, p2 - v, (255,0,0), 1, cv2.LINE_AA)

    if do_draw:
        filename = os.path.basename(path_image)
        cv2.imwrite(f'tmp/markup_check/{filename}', img)
    
    return bbox_yolo


dir_path = '/home/user/icg_data/2019_ВАСХНИЛ/Field2_2_2019'
filenames = glob.glob(f'{dir_path}/**.json')
filenames = sorted(map(os.path.basename, filenames))


# проверяем корректность разметки
for filename in filenames:
    path_markup = f'{dir_path}/{filename}'
    coordinates = read_labelme_file_to_numpy(path_markup)
    # np.savetxt(filename + '.txt', coordinates, fmt='%1.3f')

    filename    = filename.split(".")[0]
    path_image  = f'{dir_path}/{filename}.JPG'
    bbox = convert_shortcut_to_bbox(path_image, coordinates, 20, True)
    np.savetxt(f'tmp/markup_check/{filename}.txt', bbox, fmt='%1.0f')
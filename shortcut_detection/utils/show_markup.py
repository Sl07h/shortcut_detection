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


def show_markup(path_image: str, coordinates: np.ndarray) -> None:
    ''' @brief отрисовывает разметку на изображении и сохраняет его в папку tmp/markup_check
    @param path_image путь до изображения
    @param coordinates [[p0_x, p0_y, p1_x, p1_y], ...]
    '''
    img = cv2.imread(path_image)
    for coords in coordinates:
        p1 = coords[:2]
        p2 = coords[2:]
        img = cv2.line(img, p1, p2, (255,0,0), 5)
    filename = os.path.basename(path_image)
    cv2.imwrite(f'tmp/markup_check/{filename}', img)


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
    show_markup(path_image, coordinates)

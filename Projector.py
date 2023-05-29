# -*- coding: cp1251 -*-
import math
import re
import os
import glob
import time
import numpy as np
import pandas as pd
import scipy as sp
import pyvista as pv
import cv2


def get_serial(image):
    """
    Получает номер из пути (названия) изображения.
    :param image: Путь к файлу
    :return: Серийный номер
    """
    # Получим трёхзначное число из имени файла
    serial = int(re.search(r"\d{3}", os.path.basename(image)).group())

    return serial


def get_projection(path, scale=1, fill_flag=True):
    """
        Получает проекцию объекта по фотографии
        :param fill_flag: Флаг заполнения контура: при false выдаёт только границы, при true заполняет.
        :param scale: Точность изображения, при значениях выше 1 изображение уменьшается: 2 - 50%, 4 - 25%
        :param path: Путь к изображению
        :return: Массив кортежей (xl, xr, y), высоту и ширину
        """
    # Читаем изображение из файла
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Меняем размер изображения для того, чтобы уменьшить точность, но ускорить вычисления
    # Проще так, чем учитывать шаг в попиксельных вычислениях дальше
    if scale != 1:
        image = cv2.resize(image,
                           (int(image.shape[1] / scale),
                            int(image.shape[0] / scale)))

    # Создаём массив нулей как пустое изображение
    projection = np.zeros(image.shape)

    # Находим границы модели
    minY, minX = image.shape
    maxY, maxX = 0, 0

    # Построчно находим крайние пиксели и записываем в проекцию
    # Также вычисляем размеры модели на фото для нормализации
    for i in range(image.shape[0]):
        row = np.where(image[i, :] > 50)
        if len(row[0]) >= 2:
            if minY == image.shape[0]:
                minY = i
            else:
                maxY = i

            if row[0][0] < minX:
                minX = row[0][0]
            if row[0][-1] > maxX:
                maxX = row[0][-1]

            if fill_flag:
                projection[i, row[0][0]:row[0][-1]] = 255
            else:
                projection[i, row[0][0]] = 255
                projection[i, row[0][-1]] = 255

    # Нормализуем проекцию
    # TODO: Нужно реализовать вычисление максимальных границ модели и симметрично нормализовать все фотографии
    # cropped = projection[minY:maxY, minX:maxX]

    # height = maxY - minY
    # width = maxX - minX

    # cv2.imshow('image', projection)
    # cv2.waitKey(0)

    return projection, minX, minY, maxX, maxY


def get_point_cloud_cube(height, width):
    # Трёхмерный массив единиц для хранения облака точек
    # Первая координата - матрицы - X
    # Вторая координата - строки матриц - Z
    # Третья координата - элементы строк - Y
    point_cloud = np.zeros((height, width, width), dtype=int)

    return point_cloud


# shit starts from here


def get_point_projection(point, angle):
    # Перевести угол фотографии в радианы
    alpha = angle * np.pi / 180.0

    # Перевести точку в координаты
    x, z, y = point[0], point[1], point[2]

    xp = get_point_projection_x(x, z, alpha)

    return xp, z


def get_point_projection_x(x, z, alpha):
    """
    Формула вычисления проекции точки на фотографию
    :param x: х координата точки в объеме
    :param z: z координата точки в объеме
    :param alpha: угол, на котором снята фотография
    :return:
    """

    # Вычислить угол между осью х и прямой от нуля (x,z) до точки
    beta = np.arctan2(z, x) * 180.0 / np.pi

    # Координата х проекции точки на фотографию
    xp = math.sqrt(x ** 2 + z ** 2) * math.cos(beta + alpha)

    return int(xp)


def is_shaped(x, z, angle, xl, xr):
    # Перевести угол фотографии в радианы
    alpha = angle * np.pi / 180.0

    # Вычислить угол между осью х и прямой от нуля (x,z) до точки
    beta = np.arctan2(z, x) * 180.0 / np.pi

    # Координата х проекции точки на фотографию
    xp = math.sqrt(x ** 2 + z ** 2) * math.cos(beta + alpha)

    if xp < xl or xp > xr:
        return False

    return True


def make_model_from_mask(projection, angle):
    # cv2.imshow("image", projection)
    # cv2.waitKey()

    model = projection[:, :, np.newaxis]
    for i in range(projection.shape[1] - 1):
        model = np.dstack((model, projection))

    # очень плохая идея, надо переделать
    turned = sp.ndimage.interpolation.rotate(model, angle)
    H, W, D = model.shape

    turned = turned[:H, :W, :D]

    return turned


def cut_shape(point_cloud, angle, projection):
    # TODO: вырезание по формуле проекции в облако точек

    print(angle)


def save_xyz(point_cloud, path):
    file_object = open(path, 'w')

    iterator = np.nditer(point_cloud, flags=['multi_index'])

    for element in iterator:
        if element > 1:
            # x z y
            file_object.write(str(iterator.multi_index[0]) + ' ' +
                              str(iterator.multi_index[2]) + ' ' +
                              str(iterator.multi_index[1]) + '\n')

    file_object.close()


def generate_model(path, scale, radial):
    """
    Создаёт облако точек по группе изображений.
    Создаёт воксельную модель по облаку точек.
    Создаёт полигональную модель по облаку точек.
    :param path: Путь к директории с изображениями.
    :param scale: Шаг обхода по изображению в пикселях, он же размер вокселя.
    :param radial: Угол поворота камеры между изображениями.
    """

    # Получить список всех изображений в директории
    files = [f for f in glob.glob(path + "**/*.png")]

    # Переменные для размера модели
    H, W = 0, 0
    projections = {}

    # Проход по всем изображениям

    print("Получение проекций с изображений")
    for file in files:
        # Получить серийный номер фотографии из названия
        # Он же является углом поворота
        serial = get_serial(file)

        # Получить проекцию по изображению
        projection, minX, minY, maxX, maxY = get_projection(file, scale)

        projections[serial] = projection

        H, W = projection.shape
        # if maxY - minY > H:
        #     H = maxY - minY
        #
        # if maxX - minX > W:
        #     W = maxX - minX

    # Создать облако точек по максимальным размерам
    print("Создание куба облака точек")
    point_cloud = get_point_cloud_cube(H, W)

    temp = 3
    # Создать для каждого изображения трёхмерную проекцию
    for angle in projections:
        print("Вырезание проекции угла " + str(angle))
        model = make_model_from_mask(projections[angle], angle)
        print(point_cloud.shape)
        print(model.shape)
        point_cloud = np.multiply(point_cloud, model)

    print("Сохранение результата XYZ")
    xyz_path = 'point_cloud.xyz'
    save_xyz(point_cloud, xyz_path)
    pc = pv.PolyData(np.genfromtxt(xyz_path, delimiter=' ', dtype=np.float32))
    pc.plot()

    # # Создать облако точек по максимальным размерам
    # print("Создание куба облака точек")
    # point_cloud = get_point_cloud_cube(H, W)
    #
    # # Вырезать проекции объекта из облака точек
    # for angle in projections:
    #     print("Вырезание проекции угла " + str(angle))
    #     cut_shape(point_cloud, angle, projections[angle])

    # Удалить внутренности объекта

    # Сохранить результат в формате XYZ
    # print("Сохранение результата XYZ")
    # xyz_path = 'point_cloud.xyz'
    # save_xyz(point_cloud, xyz_path)

    # Вокселизировать результат

    # Создать поверхность

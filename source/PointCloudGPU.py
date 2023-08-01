# -*- coding: cp1251 -*-
import numpy
import numpy as np
import cupy as cp
import time
import pyvista as pv
import cv2

from source.Exporter import Exporter


class PointCloudGPU:
    def __init__(self):
        # self.point_cloud = np.empty((0, 3), dtype=numpy.half)
        self.point_cloud = None

    def make_point_cloud_form(self, point_cloud_20):
        # TODO: в видеопамять моего ноута на 100% не помещается даже облако из кубов размером 5 на 5

        new_cloud = cp.empty([0, 3])

        i = 100

        chunk = []

        for point in point_cloud_20:

            cube = []
            i -= 1
            x, y, z = point * 5

            # Создадим по координатам точки куб размером 5x5
            for ix in range(-2, 3):
                for iy in range(-2, 3):
                    for iz in range(-2, 3):
                        cube.append([x + ix, y + iy, z + iz])

            chunk.append(cube)

            if i == 0:
                i = 100
                new_cloud = cp.append(new_cloud, chunk)
                chunk = []

        if i != 0:
            cp.append(new_cloud, chunk)

        self.point_cloud = cp.unique(cp.array(new_cloud))

        print("something")

        # TODO: взять только уникальные точки, тем самым избавившись от дупликатов на старте

        # Exporter.save_xyz(self.point_cloud, "C:/Images/quartz_10/stretched.xyz")
        # print("ab")

    # def turn_point_cloud_horizontally(self, angle):
    #     # Угол поворота в радианах
    #     radian_angle = angle * cp.pi / 180.0
    #
    #     # Сразу вычисляем синус и косинус
    #     cos = cp.cos(radian_angle)
    #     sin = cp.sin(radian_angle)
    #
    #     # Копируем столбцы с x и y координатами
    #     x = self.point_cloud[:, 0].copy()
    #
    #     # Изменяем координаты для поворота на нужный угол
    #     self.point_cloud[:, 0] = x * cos - self.point_cloud[:, 1] * sin
    #     self.point_cloud[:, 1] = x * sin + self.point_cloud[:, 1] * cos

    # def cut_shape_from_point_cloud(self, up_down, left_right):
    #     result = []
    #
    #
    #     for left_right_border in left_right:
    #         row = self.point_cloud[cp.where(self.point_cloud[:, 2] == left_right_border[2])]
    #         row = row[cp.where(row[:, 0] > left_right_border[0])]
    #         row = row[cp.where(row[:, 0] < left_right_border[1])]
    #
    #         result.append(row)
    #
    #     self.point_cloud = cp.asarray(np.concatenate(result))

    def cut_shape_trigonometry(self, contour, angle):

        contour = cp.asarray(contour)

        result = []

        alpha = -(angle / 180.0) * cp.pi

        # self.point_cloud.argsort()

        # Уникальные z значения изображения для последующего послойного разделения
        z_values = cp.unique(self.point_cloud[:, 2])

        # Пройдём по изображению сверху вниз
        for z in z_values:
            # Возьмём строчку текущей высоты как слайс
            sl = self.point_cloud[cp.where(self.point_cloud[:, 2] == z)]

            # Получим маску этой же высоты
            mask = (contour[cp.where(contour[:, 2] == z)]).flatten()

            # Если такой маски нет, то удаляем всю строчку
            if not cp.any(mask):
                continue

            # Вытащим левую и правую границы из маски
            xl, xr, _ = mask

            # Выберем только подпадающие под граничные значения точки
            sl = sl[
                cp.where(
                    cp.sqrt(sl[:, 0] ** 2 + sl[:, 1] ** 2) * cp.cos(alpha + cp.arctan2(sl[:, 1], sl[:, 0])) > xl)]
            sl = sl[
                cp.where(
                    cp.sqrt(sl[:, 0] ** 2 + sl[:, 1] ** 2) * cp.cos(alpha + cp.arctan2(sl[:, 1], sl[:, 0])) < xr)]

            # Добавим полученный слайс к новому массиву
            result.append(sl)

        # Соединим все полученные слайсы в облако точек
        self.point_cloud = cp.concatenate(result)

    # def draw_contour_on_point_cloud(self, mask, max_width):
    #     tic = time.perf_counter()
    #     h, w = mask.shape
    #
    #     # Необходимо создать облако точек в форме контура с маски толщиной в ширину маски
    #     # TODO: этот кусок надо переписать лучше и быстрее
    #     for y in range(h):
    #
    #         frame = []
    #
    #         for x in range(w):
    #             if mask[y, x] > 0:
    #                 for depth in range(round(-max_width * 0.7 / 2), round(max_width * 0.7 / 2)):
    #                     # Переводим координаты изображения x, y в координаты облака точек, где x == x, y == z, y - глубина изображения
    #                     frame.append([x - (w / 2), depth, y - (h / 2)])
    #
    #         if len(frame) > 0:
    #             frame = cp.array(frame)
    #             self.point_cloud = cp.concatenate((self.point_cloud, frame), axis=0, dtype=numpy.half)
    #
    #     toc = time.perf_counter()
    #     diff = round(toc - tic, 3)
    #     print("PC gen: " + str(diff) + "s")
    # def viscera_disposal(self):
    #     # Массив облака точек изначально отсортирован по z, особенности генерации
    #
    #     # Пустой массив для результата
    #     result = []
    #
    #     # Возьмём индексы всех первых точек с уникальным значением высоты
    #     xy_indices = np.unique(self.point_cloud[:, 2], return_index=True)[1][1:]
    #
    #     # Разделим по этим индексам массив на подмассивы точек на одной высоте
    #     # (намного быстрее, чем каждый раз вынимать слайс из массива)
    #     sliced_array = np.split(self.point_cloud, xy_indices)
    #
    #     # Проход по каждому горизонтальному слайсу
    #     for sl in sliced_array:
    #
    #         sl = sl[sl[:, 0].argsort()]
    #
    #         # Индексы строк слайса по Х
    #         x_id = np.unique(sl[:, 0], return_index=True)[1][1:]
    #
    #         # Получим строки слайса
    #         rows = np.split(sl, x_id)
    #
    #         # Найдём минимальную и максимальную точки в строке и добавим их к результату
    #         for row in rows:
    #             try:
    #                 minX = row[np.argmin(row[:, 1], axis=0)]
    #                 result.append(minX)
    #
    #                 maxX = row[np.argmax(row[:, 1], axis=0)]
    #                 result.append(maxX)
    #             except ValueError:
    #                 continue
    #
    #         sl = sl[sl[:, 1].argsort()]
    #
    #         # Индекс столбцов слайса по Y
    #         y_id = np.unique(sl[:, 1], return_index=True)[1][1:]
    #
    #         # Получим столбцы слайса
    #         cols = np.split(sl, y_id)
    #
    #         # Найдём минимальную и максимальную точки в столбце и добавим их к результату
    #         for col in cols:
    #             try:
    #                 minY = col[np.argmin(col[:, 0], axis=0)]
    #                 result.append(minY)
    #
    #                 maxY = col[np.argmax(col[:, 0], axis=0)]
    #                 result.append(maxY)
    #             except ValueError:
    #                 continue
    #
    #     # Необходимо отсортировать облако точек по оси X для получения слайсов YZ
    #     self.point_cloud = self.point_cloud[self.point_cloud[:, 0].argsort()]
    #
    #     # Возьмём индексы всех первых точек с уникальным значением по оси X
    #     yz_indices = np.unique(self.point_cloud[:, 0], return_index=True)[1][1:]
    #
    #     # Разделим по ним предварительно отсортированный массив на подмассивы точек с одной координатой Х
    #     # Эффективно получив YZ слайсы из облака точек
    #     sliced_array = np.split(self.point_cloud, yz_indices)
    #
    #     # Проход по вертикальным слайсам YZ
    #     for sl in sliced_array:
    #
    #         sl = sl[sl[:, 1].argsort()]
    #
    #         # Индексы столбов слайса по Z
    #         y_id = np.unique(sl[:, 1], return_index=True)[1][1:]
    #
    #         # Получим столбы слайса
    #         pils = np.split(sl, y_id)
    #
    #         # Найдём минимальную и максимальную точки в строке и добавим их к результату
    #         for pil in pils:
    #             try:
    #                 minZ = pil[np.argmin(pil[:, 2], axis=0)]
    #                 result.append(minZ)
    #
    #                 maxZ = pil[np.argmax(pil[:, 2], axis=0)]
    #                 result.append(maxZ)
    #             except ValueError:
    #                 continue
    #
    #     # Возвращаем соединённые слайсы
    #     self.point_cloud = np.vstack(result)

    def apply_masks_to_point_cloud(self, images, point_cloud_20):

        self.make_point_cloud_form(point_cloud_20)

        # На каждой маске нужно
        for angle in images.left_right_masks:
            # Ноль градусов: нарисовать контур
            # Все остальные углы: повернуть матрицу, удалить точки вне маски, нарисовать контур

            print("Вырезаем под углом " + str(angle))
            # Если это не первая маска
            # if angle == 0:
            #     continue

            # print("Поворот матрицы")
            # # Повернуть матрицу
            # self.turn_point_cloud_horizontally(10)

            # print("Удаление внешних точек")
            # Удалить все точки вне матрицы
            # self.cut_shape_from_point_cloud(images.up_down_masks[angle], images.left_right_masks[angle])

            self.cut_shape_trigonometry(images.left_right_masks[angle], angle)
            # print("Построение контура")
            # # Построить контур
            # self.draw_contour_on_point_cloud(images.masks[angle], images.x1 * 0.7)

            # Exporter.save_xyz(cp.asnumpy(self.point_cloud), "C:/Images/quartz_10" + '/contour_{}.xyz'.format(angle))

        Exporter.save_xyz(cp.asnumpy(self.point_cloud), "C:/Images/quartz_10/full.xyz")

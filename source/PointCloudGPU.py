# -*- coding: cp1251 -*-
import numpy as np
import cupy as cp
import time
import pyvista as pv
import cv2

from source.Exporter import Exporter


class PointCloudGPU:
    def __init__(self):
        self.point_cloud = cp.array([[0, 0, 0]])

    def turn_point_cloud_horizontally(self, angle):
        # Угол поворота в радианах
        radian_angle = angle * cp.pi / 180.0

        # Сразу вычисляем синус и косинус
        cos = cp.cos(radian_angle)
        sin = cp.sin(radian_angle)

        # Копируем столбцы с x и y координатами
        x = self.point_cloud[:, 0]

        # Изменяем координаты для поворота на нужный угол
        self.point_cloud[:, 0] = x * cos - self.point_cloud[:, 1] * sin
        self.point_cloud[:, 1] = x * sin + self.point_cloud[:, 1] * cos

    def cut_shape_from_point_cloud(self, up_down, left_right):
        result = []
        # TODO: метод разделения облака точек на полосы через уникальные не работает
        #  на GPU, нужно придумать и написать другой

        # Вернём облако точек на CPU
        self.point_cloud = cp.asnumpy(self.point_cloud)

        # Получим индексы первых уникальных элементов массива для разделения по слайсам
        indices = np.unique(self.point_cloud[:, 2], return_index=True)[1][1:]

        # Разделять сплитом на GPU нельзя, только на CPU
        sliced_array = np.split(self.point_cloud, indices)

        for sl in sliced_array:
            try:
                z = sl[1][2]
                x = left_right[left_right[:, 2] == z]

                if x.size == 0:
                    continue

                xl, xr = x[0][0], x[0][1]

                sl = sl[xl < sl[:, 0]]
                sl = sl[sl[:, 0] < xr]

                result.append(sl)
            except IndexError:
                continue

        self.point_cloud = cp.asarray(np.concatenate(result))

    def draw_contour_on_point_cloud(self, mask, angle, max_width):
        tic = time.perf_counter()
        h, w = mask.shape

        # Необходимо создать облако точек в форме контура с маски толщиной в ширину маски
        # TODO: этот кусок надо переписать лучше и быстрее
        for y in range(h):
            for x in range(w):
                if mask[y, x] > 0:
                    frame = []
                    for depth in range(round(-max_width / 2), round(max_width / 2)):
                        # Переводим координаты изображения x, y в координаты облака точек, где x == x, y == z, y - глубина изображения
                        frame.append([x - round(w / 2), depth, y - round(h / 2)])
                    frame = cp.array(frame)

                    self.point_cloud = cp.concatenate((self.point_cloud, frame), axis=0)

        toc = time.perf_counter()
        diff = round(toc - tic, 3)
        print("PC gen: " + str(diff) + "s")

        # result = cp.empty((0, 3))

        # # Удаление всех точек снаружи маски
        # if angle != 0:
        #     indices = cp.unique(self.point_cloud[:, 2], return_index=True)[1][1:]
        #
        #     sliced_pc = cp.split(self.point_cloud, indices)
        #
        #     for sl in sliced_pc:
        #         try:
        #             y = sl[0][2]
        #             row = np.where(mask[y, :] > 0)
        #             xl = row[0][0]
        #             xr = row[0][-1]
        #             sl = cp.where(self.point_cloud[:, 0] > xl)
        #             sl = cp.where(sl[:, 0] < xr)
        #             result.append(sl)
        #
        #         except KeyError or IndexError:
        #             continue
        #
        #     self.point_cloud = cp.concatenate(result)

    def apply_masks_to_point_cloud(self, images):
        prev_angle = 0

        # На каждой маске нужно
        for angle in images.masks:
            # Ноль градусов: нарисовать контур
            # Все остальные углы: повернуть матрицу, удалить точки вне маски, нарисовать контур

            print("Вырезаем под углом " + str(angle))

            # Если это не первая маска
            if angle != 0:
                print("Поворот матрицы")
                # Повернуть матрицу
                self.turn_point_cloud_horizontally(angle - prev_angle)
                prev_angle = angle

                print("Удаление внешних точек")
                # Удалить все точки вне матрицы
                self.cut_shape_from_point_cloud(images.up_down_masks[angle], images.left_right_masks[angle])

                # Exporter.save_xyz(cp.asnumpy(self.point_cloud), "C:/Images/quartz_10" + '/contour.xyz')

            print("Построение контура")
            # Построить контур
            self.draw_contour_on_point_cloud(images.masks[angle], angle, images.x1)

        # Exporter.save_xyz(cp.asnumpy(self.point_cloud), "C:/Images/quartz_10" + '/contour.xyz')

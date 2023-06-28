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
        self.point_cloud = cp.empty((0, 3), dtype=numpy.half)

    def turn_point_cloud_horizontally(self, angle):
        # Угол поворота в радианах
        radian_angle = angle * cp.pi / 180.0

        # Сразу вычисляем синус и косинус
        cos = cp.cos(radian_angle)
        sin = cp.sin(radian_angle)

        # Копируем столбцы с x и y координатами
        x = self.point_cloud[:, 0].copy()

        # Изменяем координаты для поворота на нужный угол
        self.point_cloud[:, 0] = x * cos - self.point_cloud[:, 1] * sin
        self.point_cloud[:, 1] = x * sin + self.point_cloud[:, 1] * cos

    def cut_shape_from_point_cloud(self, up_down, left_right):
        result = []
        # TODO: метод разделения облака точек на полосы через уникальные не работает
        #  на GPU, нужно придумать и написать другой

        for left_right_border in left_right:
            row = self.point_cloud[cp.where(self.point_cloud[:, 2] == left_right_border[2])]
            row = row[cp.where(row[:, 0] > left_right_border[0])]
            row = row[cp.where(row[:, 0] < left_right_border[1])]

            result.append(row)

        self.point_cloud = cp.asarray(np.concatenate(result))

    def draw_contour_on_point_cloud(self, mask, max_width):
        tic = time.perf_counter()
        h, w = mask.shape

        # Необходимо создать облако точек в форме контура с маски толщиной в ширину маски
        # TODO: этот кусок надо переписать лучше и быстрее
        for y in range(h):

            frame = []

            for x in range(w):
                if mask[y, x] > 0:
                    for depth in range(round(-max_width * 0.7 / 2), round(max_width * 0.7 / 2)):
                        # Переводим координаты изображения x, y в координаты облака точек, где x == x, y == z, y - глубина изображения
                        frame.append([x - (w / 2), depth, y - (h / 2)])

            if len(frame) > 0:
                frame = cp.array(frame)
                self.point_cloud = cp.concatenate((self.point_cloud, frame), axis=0, dtype=numpy.half)

        toc = time.perf_counter()
        diff = round(toc - tic, 3)
        print("PC gen: " + str(diff) + "s")

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

            print("Построение контура")
            # Построить контур
            self.draw_contour_on_point_cloud(images.masks[angle], images.x1 * 0.7)

            Exporter.save_xyz(cp.asnumpy(self.point_cloud), "C:/Images/quartz_10" + '/contour_{}.xyz'.format(angle))

        # Нужно снова удалить с 0 градусов то, что нарисовалось на 360
        self.turn_point_cloud_horizontally(10)
        self.cut_shape_from_point_cloud(images.up_down_masks[0], images.left_right_masks[0])

        # # Временный костыль, надо будет отладить это
        # self.turn_point_cloud_horizontally(10)
        # prev_angle = 0
        # for angle in images.masks:
        #     # Второй проход удаления для чистки модели
        #     print("Очистка модели")
        #
        #     if angle == 190:
        #         break
        #
        #     if angle != 0:
        #         print("Поворот матрицы")
        #         # Повернуть матрицу
        #         self.turn_point_cloud_horizontally(angle - prev_angle)
        #         prev_angle = angle
        #
        #         print("Удаление внешних точек")
        #         # Удалить все точки вне матрицы
        #         self.cut_shape_from_point_cloud(images.up_down_masks[angle], images.left_right_masks[angle])

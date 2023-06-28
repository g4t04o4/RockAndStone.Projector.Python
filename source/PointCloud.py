# -*- coding: cp1251 -*-
import numpy as np
import cupy as cp
import pyvista
import time

from numba import cuda


class PointCloud:

    def __init__(self):
        self.point_cloud = None

    def make_point_cloud_cube(self, width, height):
        tic = time.perf_counter()

        height_range = np.arange(round(-height / 2), round(height / 2))
        width_range = np.arange(round(-width / 2), round(width / 2))

        self.point_cloud = np.array(np.meshgrid(width_range, width_range, height_range)).T.reshape(-1, 3)

        toc = time.perf_counter()
        diff = round(toc - tic, 3)
        print("PC gen:  " + str(diff))

    def cut_into_point_cloud(self, contour, angle):
        # Пустой массив для результата
        result = []

        # Перевод градусов в радианы
        alpha = -(angle / 180.0) * np.pi

        # Вырезание сплитом пока что самое быстрое, увеличение производительности на порядок
        # Возьмём индексы всех первых точек с уникальным значением высоты
        indices = np.unique(self.point_cloud[:, 2], return_index=True)[1][1:]

        # indices_tuple = tuple(np.atleast_1d(indices))
        # for e in indices_tuple:
        #     e = int(e)

        # Разделим по этим индексам массив на подмассивы точек на одной высоте
        # (намного быстрее, чем каждый раз вынимать слайс из массива)
        sliced_array = np.split(self.point_cloud, indices)

        # Проход по каждому горизонтальному слайсу
        for sl in sliced_array:
            sl = cp.array(sl)
            try:
                # Получим из словаря координаты граничных значений для горизонтального слайса
                xl, xr = contour[round(sl[0][2])]

                # Выберем только подпадающие под граничные значения точки
                sl = sl[
                    np.where(
                        np.sqrt(sl[:, 0] ** 2 + sl[:, 1] ** 2) * np.cos(alpha + np.arctan2(sl[:, 1], sl[:, 0])) > xl)]
                sl = sl[
                    np.where(
                        np.sqrt(sl[:, 0] ** 2 + sl[:, 1] ** 2) * np.cos(alpha + np.arctan2(sl[:, 1], sl[:, 0])) < xr)]

                # Добавим их к новому массиву
                result.append(sl)

            # На случай отсутствия в словаре значений для нужной высоты пропускаем шаг
            except KeyError or IndexError:
                continue

        # pc = pyvista.PolyData(np.concatenate(result))
        # pc.plot()

        # Возвращаем соединённые слайсы
        self.point_cloud = np.concatenate(result)

    def viscera_disposal(self):
        # Массив облака точек изначально отсортирован по z, особенности генерации

        # Пустой массив для результата
        result = []

        # Возьмём индексы всех первых точек с уникальным значением высоты
        xy_indices = np.unique(self.point_cloud[:, 2], return_index=True)[1][1:]

        # Разделим по этим индексам массив на подмассивы точек на одной высоте
        # (намного быстрее, чем каждый раз вынимать слайс из массива)
        sliced_array = np.split(self.point_cloud, xy_indices)

        # Проход по каждому горизонтальному слайсу
        for sl in sliced_array:

            sl = sl[sl[:, 0].argsort()]

            # Индексы строк слайса по Х
            x_id = np.unique(sl[:, 0], return_index=True)[1][1:]

            # Получим строки слайса
            rows = np.split(sl, x_id)

            # Найдём минимальную и максимальную точки в строке и добавим их к результату
            for row in rows:
                try:
                    minX = row[np.argmin(row[:, 1], axis=0)]
                    result.append(minX)

                    maxX = row[np.argmax(row[:, 1], axis=0)]
                    result.append(maxX)
                except ValueError:
                    continue

            sl = sl[sl[:, 1].argsort()]

            # Индекс столбцов слайса по Y
            y_id = np.unique(sl[:, 1], return_index=True)[1][1:]

            # Получим столбцы слайса
            cols = np.split(sl, y_id)

            # Найдём минимальную и максимальную точки в столбце и добавим их к результату
            for col in cols:
                try:
                    minY = col[np.argmin(col[:, 0], axis=0)]
                    result.append(minY)

                    maxY = col[np.argmax(col[:, 0], axis=0)]
                    result.append(maxY)
                except ValueError:
                    continue

        # Необходимо отсортировать облако точек по оси X для получения слайсов YZ
        self.point_cloud = self.point_cloud[self.point_cloud[:, 0].argsort()]

        # Возьмём индексы всех первых точек с уникальным значением по оси X
        yz_indices = np.unique(self.point_cloud[:, 0], return_index=True)[1][1:]

        # Разделим по ним предварительно отсортированный массив на подмассивы точек с одной координатой Х
        # Эффективно получив YZ слайсы из облака точек
        sliced_array = np.split(self.point_cloud, yz_indices)

        # Проход по вертикальным слайсам YZ
        for sl in sliced_array:

            sl = sl[sl[:, 1].argsort()]

            # Индексы столбов слайса по Z
            y_id = np.unique(sl[:, 1], return_index=True)[1][1:]

            # Получим столбы слайса
            pils = np.split(sl, y_id)

            # Найдём минимальную и максимальную точки в строке и добавим их к результату
            for pil in pils:
                try:
                    minZ = pil[np.argmin(pil[:, 2], axis=0)]
                    result.append(minZ)

                    maxZ = pil[np.argmax(pil[:, 2], axis=0)]
                    result.append(maxZ)
                except ValueError:
                    continue

        # Возвращаем соединённые слайсы
        self.point_cloud = np.vstack(result)

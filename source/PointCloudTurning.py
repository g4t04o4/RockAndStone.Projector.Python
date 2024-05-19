# -*- coding: cp1251 -*-
import numpy as np
import time

from source.Exporter import Exporter


class PointCloudTurning:
    def __init__(self, images, images_angled, damages, angle):
        self.point_cloud = None
        self.images = images
        self.images_angled = images_angled
        self.damages = damages
        self.angle = angle
        self.vertical_angle = 45

    def make_point_cloud_cube(self, width, height):
        height_range = np.arange(round(-height / 2), round(height / 2))
        width_range = np.arange(round(-width / 2), round(width / 2))

        self.point_cloud = np.array(np.meshgrid(width_range, width_range, height_range)).T.reshape(-1, 3)
        self.point_cloud = np.hstack((self.point_cloud, self.point_cloud))

    def horizontal_point_cloud_turn(self, h_angle):
        h_angle = np.deg2rad(h_angle)

        self.point_cloud[:, 0] = self.point_cloud[:, 3] * np.cos(h_angle) + self.point_cloud[:, 4] * np.sin(h_angle)
        self.point_cloud[:, 1] = - self.point_cloud[:, 3] * np.sin(h_angle) + self.point_cloud[:, 4] * np.cos(h_angle)
        self.point_cloud[:, 2] = self.point_cloud[:, 5]

    def vertical_point_cloud_turn(self, v_angle):
        v_angle = np.deg2rad(-v_angle)
        y = self.point_cloud[:, 1].copy()
        z = self.point_cloud[:, 2].copy()

        self.point_cloud[:, 0] = self.point_cloud[:, 0]
        self.point_cloud[:, 1] = y * np.cos(v_angle) + z * np.sin(v_angle)
        self.point_cloud[:, 2] = - y * np.sin(v_angle) + z * np.cos(v_angle)

    def cut_into_point_cloud(self, left_right_mask):
        # Пустой массив для результата
        result = []

        # Вырезание сплитом пока что самое быстрое, увеличение производительности на порядок
        # Возьмём индексы всех первых точек с уникальным значением высоты
        indices = np.unique(self.point_cloud[:, 2], return_index=True)[1][1:]

        # Разделим по этим индексам массив на подмассивы точек на одной высоте
        # (намного быстрее, чем каждый раз вынимать слайс из массива)
        sliced_array = np.split(self.point_cloud, indices)

        # Проход по каждому горизонтальному слайсу
        for sl in sliced_array:
            # Высота текущего слайса
            try:
                z = round(sl[0][2])
            except IndexError:
                break

            # Получим по высоте горизонтальную маску
            mask = (left_right_mask[np.where(left_right_mask[:, 2] == z)])

            # Если в маске больше трёх аргументов, то это значит, что в этом слайсе была полость и нужно поочерёдно применить все маски
            if mask.size > 3:

                for m in mask:
                    sl_copy = sl.copy()

                    # Получим из маски координаты граничных значений для горизонтального слайса
                    xl, xr, _ = m

                    # Выберем только подпадающие под граничные значения точки
                    sl_copy = sl_copy[np.where(sl_copy[:, 0] > xl)]
                    sl_copy = sl_copy[np.where(sl_copy[:, 0] < xr)]

                    # Добавим их к новому массиву
                    result.append(sl_copy)

            # Иначе просто применяем маску как обычно
            else:
                mask = mask.flatten()

                # Если маска пустая, то мы пропускаем этот слой
                if not np.any(mask):
                    continue

                # Получим из маски координаты граничных значений для горизонтального слайса
                xl, xr, _ = mask

                # Выберем только подпадающие под граничные значения точки
                sl = sl[np.where(sl[:, 0] > xl)]
                sl = sl[np.where(sl[:, 0] < xr)]

                # Добавим их к новому массиву
                result.append(sl)

            self.point_cloud = np.concatenate(result)

        # Exporter.save_xyz(self.point_cloud[:, 3:6], "C:/Images/test/test.xyz")
        # print("xyz")

    def cut_angled(self, left_right_mask):
        # Пустой массив для результата
        result = []

        # Нужно отсортировать по z, чтобы вырезание сплитом работало лучше
        self.point_cloud = self.point_cloud[self.point_cloud[:, 2].argsort()]

        # Вырезание сплитом пока что самое быстрое, увеличение производительности на порядок
        # Возьмём индексы всех первых точек с уникальным значением высоты
        indices = np.unique(self.point_cloud[:, 2], return_index=True)[1][1:]

        # Разделим по этим индексам массив на подмассивы точек на одной высоте
        # (намного быстрее, чем каждый раз вынимать слайс из массива)
        sliced_array = np.split(self.point_cloud, indices)

        # Проход по каждому горизонтальному слайсу
        for sl in sliced_array:
            # Высота текущего слайса
            try:
                z = round(sl[0][2])
            except IndexError:
                break

            # Получим по высоте горизонтальную маску
            mask = (left_right_mask[np.where(left_right_mask[:, 2] == z)])

            # Если в маске больше трёх аргументов, то это значит, что в этом слайсе была полость и нужно поочерёдно применить все маски
            if mask.size > 3:

                for m in mask:
                    sl_copy = sl.copy()

                    # Получим из маски координаты граничных значений для горизонтального слайса
                    xl, xr, _ = m

                    # Выберем только подпадающие под граничные значения точки
                    sl_copy = sl_copy[np.where(sl_copy[:, 0] > xl)]
                    sl_copy = sl_copy[np.where(sl_copy[:, 0] < xr)]

                    # Добавим их к новому массиву
                    result.append(sl_copy)

            # Иначе просто применяем маску как обычно
            else:
                mask = mask.flatten()

                # Если маска пустая, то мы пропускаем этот слой
                if not np.any(mask):
                    continue

                # Получим из маски координаты граничных значений для горизонтального слайса
                xl, xr, _ = mask

                # Выберем только подпадающие под граничные значения точки
                sl = sl[np.where(sl[:, 0] > xl)]
                sl = sl[np.where(sl[:, 0] < xr)]

                # Добавим их к новому массиву
                result.append(sl)

            self.point_cloud = np.concatenate(result)

        # Exporter.save_xyz(self.point_cloud[:, 3:6], "C:/Images/test/test.xyz")
        # print("xyz")

    def viscera_disposal(self):
        # Массив облака точек изначально отсортирован по z, особенности генерации

        # Пустой массив для результата
        result = []

        self.point_cloud = self.point_cloud[:, 3:6]

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

    def draw_damages(self, left_right_mask):
        # # Пустой массив для результата
        # result = []
        #
        # # Вырезание сплитом пока что самое быстрое, увеличение производительности на порядок
        # # Возьмём индексы всех первых точек с уникальным значением высоты
        # indices = np.unique(self.point_cloud[:, 2], return_index=True)[1][1:]
        #
        # # Разделим по этим индексам массив на подмассивы точек на одной высоте
        # # (намного быстрее, чем каждый раз вынимать слайс из массива)
        # sliced_array = np.split(self.point_cloud, indices)
        #
        # # Проход по каждому горизонтальному слайсу
        # for sl in sliced_array:
        #
        #     sl_backup = sl
        #
        #     # Высота текущего слайса
        #     try:
        #         z = round(sl[0][2])
        #     except IndexError:
        #         break
        #
        #     # Получим по высоте горизонтальную маску
        #     mask = (left_right_mask[np.where(left_right_mask[:, 2] == z)])
        #
        #     # Если в маске больше трёх аргументов, то это значит, что в этом слайсе была полость и нужно поочерёдно применить все маски
        #     if mask.size > 3:
        #
        #         for m in mask:
        #             # Получим из маски координаты граничных значений для горизонтального слайса
        #             xl, xr, _ = m
        #
        #             # Выберем только подпадающие под граничные значения точки
        #             sl = sl[np.where(sl[:, 0] > xl)]
        #             sl = sl[np.where(sl[:, 0] < xr)]
        #
        #             # Добавим их к новому массиву
        #             sl[:, 3] = 255
        #             result.append(sl)
        #             sl = sl_backup
        #
        #     # Иначе просто применяем маску как обычно
        #     else:
        #         mask = mask.flatten()
        #
        #         # Если маска пустая, то мы пропускаем этот слой
        #         if not np.any(mask):
        #             continue
        #
        #         # Получим из маски координаты граничных значений для горизонтального слайса
        #         xl, xr, _ = mask
        #
        #         # Выберем только подпадающие под граничные значения точки
        #         sl = sl[np.where(sl[:, 0] > xl)]
        #         sl = sl[np.where(sl[:, 0] < xr)]
        #
        #         # Добавим их к новому массиву
        #         sl[:, 3] = 255
        #         result.append(sl)
        #
        # self.point_cloud = (np.vstack(result))

        Exporter.save_xyzrgb(self.point_cloud, "C:/Images/test/shell.txt")

        print("xyz")

    def generate_point_cloud(self):
        # Создаём облако точек в форме куба по максимальным размерам камня на фото
        width = self.images.x1 - self.images.x0
        height = self.images.y1 - self.images.y0
        self.make_point_cloud_cube(width, height)

        # Вырезаем из облака точек по маске формы камня
        for angle in self.images.left_right_masks:
            # for angle in self.images_angled.left_right_masks:
            # Поворачиваем облако точек горизонтально перед вырезанием на нужный угол
            if angle != 0:
                self.horizontal_point_cloud_turn(angle)

            # Производим горизонтальное вырезание
            self.cut_into_point_cloud(self.images.left_right_masks[angle])

            # Exporter.save_xyz(self.point_cloud[:, 0:3], "C:/Images/test/slice_{}.xyz".format(angle))

            # Поворачиваем вертикально модель для углового вырезания
            # self.vertical_point_cloud_turn(self.vertical_angle)

            # Производим вырезание под углом
            # self.cut_angled(self.images_angled.left_right_masks[angle])

            # Exporter.save_xyz(self.point_cloud[:, 0:3], "C:/Images/test/slice_{}_angled_cut.xyz".format(angle))

        # Удаляем внутренности модели
        self.viscera_disposal()

        Exporter.save_xyz(self.point_cloud, "C:/Images/test/rock_1.xyz")

        # print("xyz")

        # TODO: Отрисовываем повреждения
        self.point_cloud = np.c_[self.point_cloud, np.zeros(self.point_cloud.shape[0])]

        for angle in self.damages.left_right_masks:
            # if angle != 0:
            #     self.horizontal_point_cloud_turn(angle)

            self.draw_damages(self.damages.left_right_masks[angle])
            break

            Exporter.save_xyzrgb(self.point_cloud, "C:/Images/test/rock_1_damages.xyz")

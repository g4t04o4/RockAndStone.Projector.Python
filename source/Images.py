# -*- coding: cp1251 -*-
import re
import os
import glob
import numpy as np
import cv2


class Images:
    def __init__(self, path, scale):

        # Путь к директории с изображениями
        self.path = path

        # Масштаб изображения
        self.scale = scale

        # Словарь для проекций
        # self.masks = {}

        # Пороговое значение
        self.intensity = 30

        # Координаты центров модели на фотографиях
        # self.center_dict = {}

        # Набор левых и правых граничных значений
        self.left_right_masks = {}

        self.left_right_masks_20 = {}

        # Набор верхних и нижних граничных значений
        self.up_down_masks = {}

        # Максимальные размеры модели
        self.x1, self.y1 = 0, 0

        # Координаты левой верхней точки модели для нормализации
        self.x0, self.y0 = 1000, 1000

    def load_gray_image(self, image_path):
        # Читаем изображение из файла
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Меняем размер изображения для того, чтобы уменьшить точность, но ускорить вычисления
        # Проще так, чем учитывать шаг в попиксельных вычислениях дальше
        if self.scale < 100:
            image = cv2.resize(image,
                               (int(image.shape[1] * self.scale / 100),
                                int(image.shape[0] * self.scale / 100)))
        return image

    # def get_contour(self, image):
    #     # Создаём пустое чёрное изображение для вывода
    #     output = np.zeros(image.shape)
    #
    #     # Получим маску по пороговому значению
    #     ret, thresh = cv2.threshold(image, self.intensity, 255, 0)
    #
    #     # Найдём в маске контуры
    #     cnt, hier = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    #     # Выберем наибольший контур
    #     c = max(cnt, key=cv2.contourArea)
    #
    #     # Нарисуем наибольший контур на пустом изображении
    #     cv2.drawContours(output, c, -1, color=255, thickness=cv2.FILLED)
    #
    #     return output

    def get_single_mask(self, file_path, angle):
        # Загружаем изображение и переводим в чёрно-белый формат
        image = self.load_gray_image(file_path)

        # Получим контур с изображения
        # contour = self.get_contour(image)

        # Находим границы модели
        minY, minX = image.shape
        maxY, maxX = 0, 0

        # TODO: Нужно получить ненормализованные горизонтальные слайсы с картинки

        # Создаём пустой массив для горизонтальных слайсов
        left_right = []

        # Высота кадра
        y = image.shape[0]

        # Проходим по строкам изображения от 0 до высоты изображения
        for i in range(y):
            # Берём из строки те пиксели, значение которых выше граничного
            row = np.where(image[i, :] > self.intensity)

            # Если в строке больше одного пикселя
            if len(row[0]) >= 2:

                # Сохраняем границы по Y
                # Если стартовое значение minY равно высоте y
                # И (!) длина этой строки белых пикселей больше или равна двум
                # Записываем текущую y координату строки
                if minY == y:
                    minY = i

                # Иначе просто на каждой непустой строке записываем координату.
                # Последняя будет максимальной
                else:
                    maxY = i

                # Вычисляем границы по X
                if row[0][0] < minX:
                    minX = row[0][0]
                if row[0][-1] > maxX:
                    maxX = row[0][-1]

                # Добавим к массиву линию как кортеж [xl, xr, y] в формате float
                left_right.append([float(row[0][0]), float(row[0][-1]), float(i)])

            #  Вычисляем максимальные размеры всей модели
            if maxX > self.x1:
                self.x1 = maxX
            if maxY > self.y1:
                self.y1 = maxY

            # Вычисляем левую верхнюю точку для последующей нормализации
            if minX < self.x0:
                self.x0 = minX
            if minY < self.y0:
                self.y0 = minY

        # TODO: получим ненормализованные вертикальные слайсы
        # Проходим по столбцам изображения

        up_down = []

        x = image.shape[1]
        for i in range(x):
            # Берём из столбца те пиксели, значение которых выше граничного
            col = np.squeeze(np.array(np.where(image[:, i] > self.intensity)))

            try:
                if len(col) >= 2:
                    # Добавим к массиву линию как кортеж [x, yu, yd]
                    up_down.append([float(i), float(col[0]), float(col[-1])])
            except TypeError:
                continue
        # Построчно находим крайние пиксели и записываем в проекцию
        # Также вычисляем размеры модели на фото для нормализации

        # Вырезаем маску под размеры модели
        # contour = contour[minY - 1:maxY + 1, minX - 1:maxX + 1]

        # Необходимо запомнить координаты центра модели относительно левого верхнего угла оригинального фото для нормализации
        centerX = round((self.x0 + self.x1) / 2)
        centerY = round((self.y0 + self.y1) / 2)

        # Нормализация
        up_down = np.array(up_down) - [centerX, centerY, centerY]
        left_right = np.array(left_right) - [centerX, centerX, centerY]

        # cv2.imshow("mask", contour)
        # cv2.waitKey()

        # Добавляем проекцию в словарь с углом в качестве ключа
        # self.masks[angle] = np.array(contour)
        self.up_down_masks[angle] = np.array(up_down)
        self.left_right_masks[angle] = np.array(left_right)

        # TODO: нужно создать маску на 20% на основе маски на 100%
        for angle in self.left_right_masks:
            lil_mask = self.left_right_masks[angle]

            lil_mask = np.round(lil_mask / 5)[0::5]

            self.left_right_masks_20[angle] = lil_mask

    def generate_masks(self):
        # Получить список всех изображений в директории
        files = [f for f in glob.glob(self.path + "**/*.png", recursive=True)]

        # Первым проходом создаём проекции и вычисляем максимально возможный размер модели
        for file_path in files:
            # Получить серийный номер фотографии из названия
            # Он же является углом поворота
            angle = int(re.search(r"\d{3}", os.path.basename(file_path)).group())

            print("Обрабатываем изображение угла " + str(angle))

            # Получить проекцию по изображению в формате набора точек границ
            # Также необходимо получить максимальные размеры формы на фотографии в пикселях
            self.get_single_mask(file_path, angle)

        # self.normalize()

        # cv2.imshow("mask", mask)
        # cv2.waitKey()

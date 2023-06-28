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
        self.masks = {}

        # Пороговое значение
        self.intensity = 25

        # Координаты центров модели на фотографиях
        self.center_dict = {}

        # Набор левых и правых граничных значений
        self.left_right_masks = {}

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

    def get_contour(self, image):
        # Создаём пустое чёрное изображение для вывода
        output = np.zeros(image.shape)

        # Получим маску по пороговому значению
        ret, thresh = cv2.threshold(image, self.intensity, 255, 0)

        # Найдём в маске контуры
        cnt, hier = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Выберем наибольший контур
        c = max(cnt, key=cv2.contourArea)

        # Нарисуем наибольший контур на пустом изображении
        cv2.drawContours(output, c, -1, color=255, thickness=cv2.FILLED)

        return output

    def get_single_mask(self, file_path, angle):
        # Загружаем изображение и переводим в чёрно-белый формат
        image = self.load_gray_image(file_path)

        # Получим контур с изображения
        contour = self.get_contour(image)

        # Находим границы модели
        minY, minX = image.shape
        maxY, maxX = 0, 0

        # TODO: Нужно получить ненормализованные горизонтальные слайсы с картинки
        # Проходим по строкам изображения

        left_right = []

        y = contour.shape[0]
        for i in range(y):
            # Берём из строки те пиксели, значение которых выше граничного
            row = np.where(contour[i, :] > 0)

            # Если в строке больше одного пикселя
            if len(row[0]) >= 2:

                # Сохраняем границы по Y
                if minY == y:
                    minY = i
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

        x = contour.shape[1]
        for i in range(x):
            # Берём из столбца те пиксели, значение которых выше граничного
            col = np.squeeze(np.array(np.where(contour[:, i] > 0)))

            try:
                if len(col) >= 2:
                    # Добавим к массиву линию как кортеж [x, yu, yd]
                    up_down.append([float(i), float(col[0]), float(col[-1])])
            except TypeError:
                continue
        # Построчно находим крайние пиксели и записываем в проекцию
        # Также вычисляем размеры модели на фото для нормализации

        # Вырезаем маску под размеры модели
        contour = contour[minY - 1:maxY + 1, minX - 1:maxX + 1]

        # Необходимо запомнить координаты центра модели относительно левого верхнего угла оригинального фото для нормализации
        self.center_dict[angle] = [round((minX + maxX) / 2), round((minY + maxY) / 2)]

        # cv2.imshow("mask", contour)
        # cv2.waitKey()

        # Добавляем проекцию в словарь с углом в качестве ключа
        self.masks[angle] = np.array(contour)
        self.up_down_masks[angle] = np.array(up_down)
        self.left_right_masks[angle] = np.array(left_right)

    def normalize(self):
        # TODO: Необходимо нормализовать контуры

        for angle in self.center_dict:
            self.up_down_masks[angle] -= [self.center_dict[angle][0],
                                          self.center_dict[angle][1],
                                          self.center_dict[angle][1]]

            self.left_right_masks[angle] -= [self.center_dict[angle][0] - 1,
                                             self.center_dict[angle][0] + 1,
                                             self.center_dict[angle][1]]

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

        self.normalize()

        # cv2.imshow("mask", mask)
        # cv2.waitKey()

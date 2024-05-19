# -*- coding: cp1251 -*-
from glob import glob
import numpy as np
import cv2


class Images:
    def __init__(self, path, scale, angle, angled_flag):

        # Путь к директории с изображениями
        self.path = path

        # Масштаб изображения
        self.scale = scale

        # Угол поворота камеры на изображениях
        self.angle = angle

        # Пороговое значение яркости
        self.intensity = 100

        # Набор обработанных изображений с контурами
        self.contours = {}

        # Набор левых и правых граничных значений
        self.left_right_masks = {}

        # Максимальные размеры модели
        self.x1, self.y1 = 0, 0

        # Координаты левой верхней точки модели для нормализации
        self.x0, self.y0 = 2000, 2000

        self.angled_flag = angled_flag

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

    def get_single_contour(self, file_path, angle):
        # загрузка чёрно-белого изображения
        image = self.load_gray_image(file_path)

        # вырезание рабочего кадра
        # if self.angled_flag == 1:
        #     h, w = image.shape
        #     h0, h1 = int(1 * h / 5), int(3 * h / 5)
        #     w0, w1 = int(w / 4), int(3 * w / 4)
        #     image = image[h0:h1, w0:w1]
        # else:
        #     h, w = image.shape
        #     h0, h1 = int(3 * h / 5), int(h)
        #
        #     w0, w1 = int(w / 4), int(3 * w / 4)
        #     image = image[h0:h1, w0:w1]

        # cv2.imshow("image", image)
        # cv2.waitKey()

        ret, thresh = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)

        # cv2.imshow("image", thresh)
        # cv2.waitKey()

        # коэффициент ядра для морфологических операций над фотографией, зависит от масштаба фото
        k = 12 * self.scale // 100
        # kernel = np.ones((k, k), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        # thresh = cv2.erode(thresh, kernel, iterations=1)

        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest = sorted_contours[0]

        black = np.zeros(image.shape)
        cv2.drawContours(black, largest, -1, (255, 0, 0), thickness=1)
        cv2.fillPoly(black, pts=[largest], color=255)

        # cv2.imshow("image", black)
        # cv2.waitKey()

        black = cv2.erode(black, kernel, iterations=1)
        black = cv2.erode(black, kernel, iterations=1)
        black = cv2.dilate(black, kernel, iterations=1)

        # cv2.imshow("image", black)
        # cv2.waitKey()

        self.contours[angle] = black

    def get_single_mask(self, angle):
        # Загружаем полученный контур
        image = self.contours[angle]

        # Находим границы модели
        minY, minX = image.shape
        maxY, maxX = 0, 0

        # Создаём пустой массив для горизонтальных слайсов
        left_right = []

        # Высота кадра
        y = image.shape[0]

        # Проходим по строкам изображения от 0 до высоты изображения
        for i in range(y):

            row_data = image[i, :]

            # Берём из строки те пиксели, значение которых выше граничного
            row = np.where(row_data > self.intensity)

            # Если в строке больше одного пикселя
            if len(row[0]) > 1:

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

                # TODO: нужно пропускать полости меньше 10 пикселей

                left_border = row[0][0]
                right_border = row[0][-1]

                temp_row = row_data[row[0][0]:row[0][-1]]

                # Получим только те биты, которые попадают под граничные условия
                if temp_row[temp_row == 0.0].size == 0:
                    left_right.append([float(left_border), float(right_border), float(i)])
                else:

                    while True:
                        try:
                            # Заполним единицами все пиксели левее левой границы
                            row_data[0:int(left_border)] = 255.0
                            # Найдём координаты первого встречного нуля - координаты начала полости
                            left_cavity_border = np.argwhere(row_data == 0.0)[0] - 1
                            # Сохраним в массив масок первый кусок камня
                            left_right.append([float(left_border), float(left_cavity_border), float(i)])
                            # Заполним нулями сохранённый кусок
                            row_data[0:int(left_cavity_border + 1)] = 0.0

                            # Если в оставшейся строке ещё остался кусок камня
                            if row_data[row_data == 255.0].size != 0:
                                left_border = np.argwhere(row_data == 255.0)[0]
                            else:
                                break
                        except IndexError:
                            break

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

        # Необходимо запомнить координаты центра модели относительно левого верхнего угла оригинального фото для нормализации
        centerX = round((self.x0 + self.x1) / 2)
        centerY = round((self.y0 + self.y1) / 2)

        # Нормализация
        left_right = np.array(left_right) - [centerX, centerX, centerY]

        # Добавляем проекцию в словарь с углом в качестве ключа
        self.left_right_masks[angle] = np.array(left_right)

    def generate_masks(self):
        # Получить список всех изображений в директории
        files = []
        for ext in ("**/*.jpg", "**/*.png", "**/*.bmp"):
            files.extend(glob(self.path + ext, recursive=True))

        curr_angle = 0.0

        # Первым проходом создаём проекции и вычисляем максимально возможный размер модели
        for file_path in files:
            # Получим контуры по изображениям и вычислим максимальные размеры кристалла
            self.get_single_contour(file_path, curr_angle)

            # Нормализуем все изображения под размер кристалла и снимем с них маски
            self.get_single_mask(curr_angle)

            curr_angle += self.angle

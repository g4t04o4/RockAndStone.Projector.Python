# -*- coding: cp1251 -*-
import re
import os
import glob
import time
import numpy as np
import pyvista as pv
import cv2
import pymeshfix as pmf
import PVGeo as pvg


def get_serial(image):
    """
    Получает номер из пути (названия) изображения.
    :param image: Путь к файлу
    :return: Серийный номер
    """
    # Получим трёхзначное число из имени файла
    serial = int(re.search(r"\d{3}", os.path.basename(image)).group())

    return serial


def get_projection(path, scale=100):
    """
    Получает проекцию объекта по фотографии в формате набора координат левой и правой границ (xl, xr, y)
    :param path: Путь к изображению
    :param scale: Масштабирование изображения в процентах для понижения точности, не ставить больше 100
    :return: Массив кортежей (xl, xr, y), высоту и ширину, а также координаты верхней левой точки квадрата
    """
    # Читаем изображение из файла
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Меняем размер изображения для того, чтобы уменьшить точность, но ускорить вычисления
    # Проще так, чем учитывать шаг в попиксельных вычислениях дальше
    if scale < 100:
        image = cv2.resize(image,
                           (int(image.shape[1] * scale / 100),
                            int(image.shape[0] * scale / 100)))

    # Создаём массив для хранения координат (не numpy)
    lines = []

    # Находим границы модели
    minY, minX = image.shape
    maxY, maxX = 0, 0

    # Построчно находим крайние пиксели и записываем в проекцию
    # Также вычисляем размеры модели на фото для нормализации
    # Проходим по строкам изображения
    for i in range(image.shape[0]):
        # Если значение intensity пикселя больше 50
        row = np.where(image[i, :] > 50)
        if len(row[0]) >= 2:

            # Сохраняем границы по Y
            if minY == image.shape[0]:
                minY = i
            else:
                maxY = i

            # Вычисляем границы по X
            if row[0][0] < minX:
                minX = row[0][0]
            if row[0][-1] > maxX:
                maxX = row[0][-1]

            # Добавим к массиву проекции линию как кортеж [xl, xr, y]
            lines.append([float(row[0][0]), float(row[0][-1]), float(i)])

    # Вычисляем размеры модели по границам
    height = maxY - minY
    width = maxX - minX

    # Переведём массив питона в numpy для ускорения последующих вычислений
    np_lines = np.array(lines)

    # cv2.imshow("image", image)
    # cv2.waitKey()

    # print("Размеры: {}, {}, {}, {}".format(minX, minY, height, width))

    return np_lines, height, width, minX, minY


def get_projections_from_files(path, scale):
    """
    Получить словарь с проекциями
    :param path: Путь к директории с изображениями
    :param scale: Масштаб изображения в процентах
    :return: Возвращает словарь проекций, а также максимальные размеры модели и координаты центра модели для нормализации
    """
    # Получить список всех изображений в директории
    files = [f for f in glob.glob(path + "**/*.png")]

    # Словарь для проекций
    projections = {}

    # Максимальные размеры модели
    max_width, max_height = 0, 0

    # Координаты левой верхней точки модели для нормализации
    norm_X, norm_Y = 1000, 1000

    # Первым проходом создаём проекции и вычисляем максимально возможный размер модели
    print("Получение проекций с изображений")
    for file in files:
        # Получить серийный номер фотографии из названия
        # Он же является углом поворота
        serial = get_serial(file)

        # Получить проекцию по изображению в формате набора точек границ
        # Также необходимо получить максимальные размеры формы на фотографии в пикселях
        projection, height, width, x0, y0 = get_projection(file, scale)

        # Вычисляем максимальные размеры всей модели
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height

        # Вычисляем левую верхнюю точку для последующей нормализации
        if x0 < norm_X:
            norm_X = x0
        if y0 < norm_Y:
            norm_Y = y0

        # Добавляем проекцию в словарь с углом в качестве ключа
        projections[serial] = np.array(projection)

    # Нужно поставить центр системы координат в центр изображения
    norm_X += round(max_width / 2)
    norm_Y += round(max_height / 2)

    # print("Размеры модели: {}, {}, {}, {}".format(max_width, max_height, norm_X, norm_Y))

    return projections, max_width, max_height, norm_X, norm_Y


def make_point_cloud_cube(max_height, max_width):
    """
    Создаёт облако точек под максимальные размеры в формате матрицы x, y, z с центром системы координат в центре куба
    :param max_height: Максимальная высота модели
    :param max_width: Максимальная ширина модели
    :return: Облако точек в форме куба в формате [x, y, z]
    """
    print("Создание облака точек {} x {} x {}".format(max_width, max_width, max_height))

    h_range = np.arange(-max_height / 2, max_height / 2)
    w_range = np.arange(-max_width / 2, max_width / 2)

    point_cloud = np.array(np.meshgrid(w_range, w_range, h_range)).T.reshape(-1, 3)

    # Нет умножения на матрицу проекции или подсчёта проекций, надо будет сильно переделать
    # print(point_cloud.shape[0])
    #
    # zer = np.zeros((point_cloud.shape[0], 1))
    #
    # point_cloud = np.hstack((point_cloud, zer))

    # # Пустой массив для хранения облака
    # print("Создание облака точек")
    # point_cloud = []
    #
    # # Проход по размерам потенциального облака
    # for x in range(max_width):
    #     for y in range(max_width):
    #         for z in range(max_height):
    #             # В каждой итерации записываем новую точку в облако
    #             point_cloud.append([x - max_width / 2, y - max_width / 2, z - max_height / 2])
    #
    # # Возвращаем массив в формате numpy для оптимизации последующих вычислений
    # return np.array(point_cloud)

    return point_cloud


def normalize(projection, nx, ny):
    """
    Возвращает нормализованную проекцию
    :param projection: Проекция по изображению
    :param nx: Коэффициент нормализации Х
    :param ny: Коэффициент нормализации Y
    :return: Нормализованная проекция
    """

    # Использовать размеры модели для нормализации всех проекций под один размер
    # Добавляя к координатам половину разницы между размером и максимальным размером, мы центрируем модель
    # Вычитаем из всей проекции координаты первой точки, чтобы быстро получить нормализованную матрицу координат
    projection -= [nx, nx, ny]

    # Перевести в словарь
    lines = {}
    for line in projection:
        lines[round(line[2])] = [line[0], line[1]]

    return lines

    return projection


def turn_point_cloud_horizontally(point_cloud, angle):
    """
    Поворот облака точек по горизонтали (относительно оси Z) на определённый угол
    :param point_cloud: Облако точек, которое требуется повернуть
    :param angle: Угол, на который требуется повернуть облако точек
    :return: Трансформированное облако точек
    """
    # Угол поворота в радианах
    radian_angle = angle * np.pi / 180.0

    # Сразу вычисляем синус и косинус
    cos = np.cos(radian_angle)
    sin = np.sin(radian_angle)

    # Копируем столбцы с x и y координатами
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]

    # Изменяем координаты для поворота на нужный угол
    point_cloud[:, 0] = x * cos - y * sin
    point_cloud[:, 1] = x * sin + y * cos

    return point_cloud


def cut_into_point_cloud(point_cloud, lines):
    # Пустой массив для результата
    result = []

    # Вырезание сплитом пока что самое быстрое, увеличение производительности на порядок
    # Возьмём индексы всех первых точек с уникальным значением высоты
    indices = np.unique(point_cloud[:, 2], return_index=True)[1][1:]

    # Разделим по этим индексам массив на подмассивы точек на одной высоте
    # (намного быстрее, чем каждый раз вынимать слайс из массива)
    sliced_array = np.split(point_cloud, indices)

    # Проход по каждому горизонтальному слайсу
    for sl in sliced_array:
        try:
            # Получим из словаря координаты граничных значений для горизонтального слайса
            # y = round(sl[0][2])
            xl, xr = lines[round(sl[0][2])]

            # sl[:3] = np.sqrt(np.sl[:0] ** 2 + sl[:1] ** 2)

            # Выберем только подпадающие под граничные значения точки
            sl = sl[xl < sl[:, 0]]
            sl = sl[sl[:, 0] < xr]

            # Добавим их к новому массиву
            result.append(sl)

        # На случай отсутствия в словаре значений для нужной высоты пропускаем шаг
        except KeyError:
            continue

    # Возвращаем соединённые слайсы
    return np.concatenate(result)


def save_xyz(point_cloud, path):
    """
    Сохраняем облако точек в формате XYZ
    :param point_cloud: Облако точек
    :param path: Путь для сохранения файла XYZ
    """
    print("Сохранение результата XYZ")

    file_object = open(path, 'w')

    for point in point_cloud:
        file_object.write(str(point[0]) + ' ' +
                          str(point[1]) + ' ' +
                          str(point[2]) + '\n')

    file_object.close()


def viscera_disposal(point_cloud):
    print("Вырезание внутренностей")
    # Массив облака точек изначально отсортирован по z, особенности генерации

    # Пустой массив для результата
    result = []

    # Возьмём индексы всех первых точек с уникальным значением высоты
    xy_indices = np.unique(point_cloud[:, 2], return_index=True)[1][1:]

    # Разделим по этим индексам массив на подмассивы точек на одной высоте
    # (намного быстрее, чем каждый раз вынимать слайс из массива)
    sliced_array = np.split(point_cloud, xy_indices)

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
    point_cloud = point_cloud[point_cloud[:, 0].argsort()]

    # Возьмём индексы всех первых точек с уникальным значением по оси X
    yz_indices = np.unique(point_cloud[:, 0], return_index=True)[1][1:]

    # Разделим по ним предварительно отсортированный массив на подмассивы точек с одной координатой Х
    # Эффективно получив YZ слайсы из облака точек
    sliced_array = np.split(point_cloud, yz_indices)

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
    return np.vstack(result)


def voxelize_point_cloud(point_cloud):
    grid = pvg.filters.VoxelizePoints().apply(point_cloud)
    p = pv.Plotter(notebook=0)
    p.add_mesh(grid, opacity=0.5, show_edges=True)
    p.add_mesh(point_cloud, point_size=5, color="red")
    p.show_grid()
    p.show()


def generate_surface(point_cloud, neighbours, scale):
    print("Генерация поверхности с количеством соседних точек " + str(neighbours))
    mesh = pv.PolyData(point_cloud).reconstruct_surface(nbr_sz=neighbours)
    fixed = pmf.MeshFix(mesh)
    fixed.repair()
    fixed.mesh.save("mesh_{}.stl".format(scale))
    fixed.mesh.plot()


def generate_model(path, scale, radial):
    """
    Основной метод, генерирующий модели с изображений
    :param path: Путь к директории с изображениями
    :param scale: Коэффициент сжатия изображения в процентах
    :param radial: Угол поворота камеры на изображениях
    :return: Метод ничего не возвращает, но сохраняет файлы моделей в директорию с изображениями
    """
    g_tic = time.perf_counter()  # Глобальный счётчик времени

    # Получим проекции с изображений, а также максимальные размеры модели и левую верхнюю точку для нормализации
    projections, max_width, max_height, norm_X, norm_Y = get_projections_from_files(path, scale)

    # Создать облако точек под максимальные размеры в формате матрицы x, y, z
    tic = time.perf_counter()
    point_cloud = make_point_cloud_cube(max_height, max_width)
    toc = time.perf_counter()
    print(round(toc - tic, 3))

    # Вторым проходом нормализуем все проекции и начинаем вырезать облако
    for angle in projections:
        print("Вырезание формы " + str(angle) + " из облака точек")

        tic = time.perf_counter()

        # Нормализация проекции, возвращает словарь нормализованных линий
        lines = normalize(projections[angle], norm_X, norm_Y)

        # Горизонтальный поворот модели
        if angle != 0:
            point_cloud = turn_point_cloud_horizontally(point_cloud, radial)

        # Удаляем из матрицы все точки, не принадлежащие проекции
        point_cloud = cut_into_point_cloud(point_cloud, lines)

        toc = time.perf_counter()
        print(round(toc - tic, 3))

        # Выводим результат вырезания
        # pv.plot(pv.PolyData(point_cloud))

    # Выводим результат
    pv.plot(pv.PolyData(point_cloud))

    # Удалить внутренности облака точек
    # Вместо удаления внутренностей можно брать первую точку с каждой из сторон, как будто берём слепок с облака точек
    tic = time.perf_counter()
    point_cloud = viscera_disposal(point_cloud)
    toc = time.perf_counter()
    print(round(toc - tic, 3))

    # Вывод общего времени вычисления без вывода
    print("Общее время вычисления")
    g_toc = time.perf_counter()
    print(round(g_toc - g_tic, 3))

    # Выводим результат
    pc = pv.PolyData(point_cloud)
    pv.plot(pc)

    # Сохранить облако точек
    tic = time.perf_counter()  # Счётчик сохранения облака точек в файл
    save_xyz(point_cloud, 'point_cloud_{}.xyz'.format(scale))
    toc = time.perf_counter()
    print(round(toc - tic, 3))

    # Вокселизировать облако точек
    # Не работает на больших значениях
    # voxelize_point_cloud(point_cloud)

    # Создать поверхность облака точек
    tic = time.perf_counter()
    generate_surface(pc, 17, scale)
    toc = time.perf_counter()
    print(round(toc - tic, 3))

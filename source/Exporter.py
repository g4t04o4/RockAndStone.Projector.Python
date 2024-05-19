# -*- coding: cp1251 -*-
import time
import numpy as np


class Exporter:
    @staticmethod
    def load_xyz_point_cloud(path):
        points = np.genfromtxt(path, delimiter=' ', dtype=np.float32)
        return points

    @staticmethod
    def save_xyz(point_cloud, path):
        tic = time.perf_counter()

        print("Сохранение результата XYZ")

        with open(path, 'w+') as file_object:
            for point in point_cloud:
                file_object.write(str(point[0]) + ' ' +
                                  str(point[1]) + ' ' +
                                  str(point[2]) + '\n')

        toc = time.perf_counter()
        print(round(toc - tic, 3))

    @staticmethod
    def save_xyzrgb(point_cloud, path):
        tic = time.perf_counter()
        damage_color = '255 0 0'

        print("Сохранение результата XYZRGB")

        with open(path, 'w+') as file_object:
            for point in point_cloud:
                if point[3] != 0:
                    # Точка записывается как повреждённая
                    file_object.write(str(point[0]) + ' ' +
                                      str(point[1]) + ' ' +
                                      str(point[2]) + ' ' +
                                      damage_color + '\n')
                else:
                    # Точка записывается как неповреждённая
                    file_object.write(str(point[0]) + ' ' +
                                      str(point[1]) + ' ' +
                                      str(point[2]) + '\n')

        toc = time.perf_counter()
        print(round(toc - tic, 3))

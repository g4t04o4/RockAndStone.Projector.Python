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
    def save_xyz_gpu(point_cloud, path):
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

        print("Сохранение результата XYZRGB")

        with open(path, 'w+') as file_object:
            for point in point_cloud:
                if point[3] != 0:
                    file_object.write(str(point[0]) + ' ' +
                                      str(point[1]) + ' ' +
                                      str(point[2]) + ' 0 191 255\n')
                else:
                    file_object.write(str(point[0]) + ' ' +
                                      str(point[1]) + ' ' +
                                      str(point[2]) + ' 176 196 222\n')

        toc = time.perf_counter()
        print(round(toc - tic, 3))

    def convert_to_hhmmyy(time_in_seconds):
        if time_in_seconds < 59:
            return (time_in_seconds)
        elif time_in_seconds < 3599:
            return "{}:{}".format(round(time_in_seconds // 60),
                                  round(time_in_seconds % 60, 3))
        else:
            return "{}:{}:{}".format(round(time_in_seconds // 3600),
                                     round((time_in_seconds % 3600) // 60),
                                     round((time_in_seconds % 3600) % 60), 3)

    def get_time(path, scale, pc_time, vx_time, sf_time, neighbours):
        with open(path, 'w+') as file_object:
            out_str = "Scale: {}%\n" \
                      "\tPoint cloud: {}s\n" \
                      "\tVoxelization: {}s\n" \
                      "\tSurface recon: {}s\n" \
                      "\t(with {} neighbours)".format(scale,
                                                      convert_to_hhmmyy(pc_time),
                                                      convert_to_hhmmyy(vx_time),
                                                      convert_to_hhmmyy(sf_time),
                                                      neighbours)
            file_object.write(out_str)

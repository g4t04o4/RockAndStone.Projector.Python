# -*- coding: cp1251 -*-
import time
import numpy as np
import cupy as cp

from source.Images import Images
from source.PointCloudGPU import PointCloudGPU
from source.PointCloud import PointCloud
from source.Exporter import Exporter


class Projector:

    def __init__(self, path):
        self.path = path
        self.images_20 = Images(path, 20)
        self.images = Images(path, 100)
        self.point_cloud_20 = PointCloud()
        self.point_cloud = PointCloudGPU()

    def generate_model(self):
        # print("Генерация контуров по изображениям 20% масштаба")
        self.images_20.generate_masks()

        print("Генерация контуров в полном масштабе")
        self.images.generate_masks()

        # TODO: можно попробовать сгенерировать облако точек на 20 на масках от 100%, разделённых на 5,
        #  таким образом они будут точно одинаково центрированы
        print("Генерация облака точек на 20% масштабе")
        self.point_cloud_20.generate_point_cloud(self.images_20)

        print("Сохранение облака точек на 20% масштабе")
        Exporter.save_xyz(self.point_cloud_20.point_cloud, self.path + '/point_cloud_20.xyz')

        print("Генерация облака точек в полном масштабе")
        self.point_cloud.apply_masks_to_point_cloud(self.images, self.point_cloud_20.point_cloud)

        print("Сохранение облака точек")
        Exporter.save_xyz(cp.asnumpy(self.point_cloud.point_cloud), self.path + '/point_cloud_100.xyz')

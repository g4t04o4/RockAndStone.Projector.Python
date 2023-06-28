# -*- coding: cp1251 -*-
import time
import numpy as np
import cupy as cp

from source.Images import Images
from source.PointCloudGPU import PointCloudGPU
from source.Exporter import Exporter


class Projector:

    def __init__(self, path, scale):
        self.path = path
        self.images = Images(path, scale)
        self.point_cloud = PointCloudGPU()

    def generate_model(self):
        print("Генерация контуров по изображениям")
        self.images.generate_masks()

        print("Генерация облака точек")
        self.point_cloud.apply_masks_to_point_cloud(self.images)

        print("Сохранение облака точек")
        Exporter.save_xyz(cp.asnumpy(self.point_cloud.point_cloud), self.path + '/point_cloud.xyz')

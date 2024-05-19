# -*- coding: cp1251 -*-
import time

from source.Images import Images
# from source.PointCloud import PointCloud
from source.PointCloudTurning import PointCloudTurning as PointCloud
from source.Exporter import Exporter


class Projector:

    def __init__(self, image_path, angled_path, damage_path, scale, angle):
        self.path = image_path
        self.angled_path = angled_path
        self.damage_path = damage_path
        self.scale = scale
        self.angle = angle
        self.images = Images(self.path, self.scale, self.angle, angled_flag=0)
        self.images_angled = Images(self.angled_path, self.scale, self.angle, angled_flag=1)
        self.damages = Images(self.path, self.scale, self.angle, angled_flag=0)
        self.point_cloud = PointCloud(self.images, self.images_angled, self.damages, self.angle)

    def generate_model(self):
        print("Генерируем {}% маски...".format(self.scale))
        tic = time.perf_counter()

        self.images.generate_masks()
        self.images_angled.generate_masks()
        self.damages.generate_masks()

        toc = time.perf_counter()
        print("Маски на {}% сгенерировались за:\n {}s".format(self.scale, round(toc - tic, 3)))

        print("Генерируем облако точек...")
        tic = time.perf_counter()

        self.point_cloud.generate_point_cloud()

        toc = time.perf_counter()
        print("Облако точек сгенерировалось за:\n {}s".format(round(toc - tic, 3)))

        # TODO: нужно загрузить маску и отрисовать её на камне

        Exporter.save_xyz(self.point_cloud.point_cloud, self.path + "/point_cloud_{}.xyz".format(self.scale))

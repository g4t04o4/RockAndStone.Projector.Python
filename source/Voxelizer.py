# -*- coding: cp1251 -*-
import re
import os
import glob
import time
import numpy as np
import pyvista as pv
import cv2
import pymeshfix as pmf

cube = [
    [[0, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
    [[0, 0, 1], [0, 1, 1], [1, 0, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1]],

    [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
    [[1, 1, 1], [1, 0, 1], [0, 1, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0]],

    [[0, 1, 1], [0, 1, 0], [1, 1, 0]],
    [[0, 1, 1], [1, 1, 1], [1, 1, 0]],
    [[1, 1, 0], [1, 0, 0], [1, 0, 1]],
    [[1, 1, 0], [1, 1, 1], [1, 0, 1]]
]


def get_facet(x, y, z, facet):
    facet_string = "\tfacet normal 0 0 0\n\t\touter loop\n"

    for facet_point in facet:
        facet_string += "\t\t\tvertex {} {} {}\n".format(x + facet_point[0], y + facet_point[1], z + facet_point[2])

    facet_string += "\t\tendloop\n\tendfacet\n"

    return facet_string


def get_voxel(point):
    voxel_string = ""

    # Куб состоит из двенадцати треугольных граней, выраженных через полигоны
    # Каждый полигон состоит из трёх точек
    for facet in cube:
        x, y, z = point
        voxel_string += get_facet(x, y, z, facet)

    return voxel_string


def generate_voxels(point_cloud, path):
    print("Вокселизация")

    list_cloud = list(point_cloud)

    with open(path, 'w+', buffering=1024) as file_object:
        file_object.write("solid model\n")

        for point in list_cloud:
            file_object.write(get_voxel(point))

        file_object.write("endsolid model")

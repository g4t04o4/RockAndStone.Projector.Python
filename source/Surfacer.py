# -*- coding: cp1251 -*-
import re
import os
import glob
import time
import numpy as np
import pyvista as pv
import cv2
import pymeshfix as pmf


def generate_surface(point_cloud, neighbours, path):
    print("Генерация поверхности с количеством соседних точек " + str(neighbours))
    mesh = pv.PolyData(point_cloud).reconstruct_surface(nbr_sz=neighbours)
    fixed = pmf.MeshFix(mesh)
    fixed.repair()
    fixed.mesh.save(path)

# -*- coding: cp1251 -*-
import math
import re
import os
import glob
import time
import numpy as np
import pandas as pd
import scipy as sp
import pyvista as pv
import cv2


def get_serial(image):
    """
    �������� ����� �� ���� (��������) �����������.
    :param image: ���� � �����
    :return: �������� �����
    """
    # ������� ���������� ����� �� ����� �����
    serial = int(re.search(r"\d{3}", os.path.basename(image)).group())

    return serial


def get_projection(path, scale=1, fill_flag=True):
    """
        �������� �������� ������� �� ����������
        :param fill_flag: ���� ���������� �������: ��� false ����� ������ �������, ��� true ���������.
        :param scale: �������� �����������, ��� ��������� ���� 1 ����������� �����������: 2 - 50%, 4 - 25%
        :param path: ���� � �����������
        :return: ������ �������� (xl, xr, y), ������ � ������
        """
    # ������ ����������� �� �����
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # ������ ������ ����������� ��� ����, ����� ��������� ��������, �� �������� ����������
    # ����� ���, ��� ��������� ��� � ������������ ����������� ������
    if scale != 1:
        image = cv2.resize(image,
                           (int(image.shape[1] / scale),
                            int(image.shape[0] / scale)))

    # ������ ������ ����� ��� ������ �����������
    projection = np.zeros(image.shape)

    # ������� ������� ������
    minY, minX = image.shape
    maxY, maxX = 0, 0

    # ��������� ������� ������� ������� � ���������� � ��������
    # ����� ��������� ������� ������ �� ���� ��� ������������
    for i in range(image.shape[0]):
        row = np.where(image[i, :] > 50)
        if len(row[0]) >= 2:
            if minY == image.shape[0]:
                minY = i
            else:
                maxY = i

            if row[0][0] < minX:
                minX = row[0][0]
            if row[0][-1] > maxX:
                maxX = row[0][-1]

            if fill_flag:
                projection[i, row[0][0]:row[0][-1]] = 255
            else:
                projection[i, row[0][0]] = 255
                projection[i, row[0][-1]] = 255

    # ����������� ��������
    # TODO: ����� ����������� ���������� ������������ ������ ������ � ����������� ������������� ��� ����������
    # cropped = projection[minY:maxY, minX:maxX]

    # height = maxY - minY
    # width = maxX - minX

    # cv2.imshow('image', projection)
    # cv2.waitKey(0)

    return projection, minX, minY, maxX, maxY


def get_point_cloud_cube(height, width):
    # ��������� ������ ������ ��� �������� ������ �����
    # ������ ���������� - ������� - X
    # ������ ���������� - ������ ������ - Z
    # ������ ���������� - �������� ����� - Y
    point_cloud = np.zeros((height, width, width), dtype=int)

    return point_cloud


# shit starts from here


def get_point_projection(point, angle):
    # ��������� ���� ���������� � �������
    alpha = angle * np.pi / 180.0

    # ��������� ����� � ����������
    x, z, y = point[0], point[1], point[2]

    xp = get_point_projection_x(x, z, alpha)

    return xp, z


def get_point_projection_x(x, z, alpha):
    """
    ������� ���������� �������� ����� �� ����������
    :param x: � ���������� ����� � ������
    :param z: z ���������� ����� � ������
    :param alpha: ����, �� ������� ����� ����������
    :return:
    """

    # ��������� ���� ����� ���� � � ������ �� ���� (x,z) �� �����
    beta = np.arctan2(z, x) * 180.0 / np.pi

    # ���������� � �������� ����� �� ����������
    xp = math.sqrt(x ** 2 + z ** 2) * math.cos(beta + alpha)

    return int(xp)


def is_shaped(x, z, angle, xl, xr):
    # ��������� ���� ���������� � �������
    alpha = angle * np.pi / 180.0

    # ��������� ���� ����� ���� � � ������ �� ���� (x,z) �� �����
    beta = np.arctan2(z, x) * 180.0 / np.pi

    # ���������� � �������� ����� �� ����������
    xp = math.sqrt(x ** 2 + z ** 2) * math.cos(beta + alpha)

    if xp < xl or xp > xr:
        return False

    return True


def make_model_from_mask(projection, angle):
    # cv2.imshow("image", projection)
    # cv2.waitKey()

    model = projection[:, :, np.newaxis]
    for i in range(projection.shape[1] - 1):
        model = np.dstack((model, projection))

    # ����� ������ ����, ���� ����������
    turned = sp.ndimage.interpolation.rotate(model, angle)
    H, W, D = model.shape

    turned = turned[:H, :W, :D]

    return turned


def cut_shape(point_cloud, angle, projection):
    # TODO: ��������� �� ������� �������� � ������ �����

    print(angle)


def save_xyz(point_cloud, path):
    file_object = open(path, 'w')

    iterator = np.nditer(point_cloud, flags=['multi_index'])

    for element in iterator:
        if element > 1:
            # x z y
            file_object.write(str(iterator.multi_index[0]) + ' ' +
                              str(iterator.multi_index[2]) + ' ' +
                              str(iterator.multi_index[1]) + '\n')

    file_object.close()


def generate_model(path, scale, radial):
    """
    ������ ������ ����� �� ������ �����������.
    ������ ���������� ������ �� ������ �����.
    ������ ������������� ������ �� ������ �����.
    :param path: ���� � ���������� � �������������.
    :param scale: ��� ������ �� ����������� � ��������, �� �� ������ �������.
    :param radial: ���� �������� ������ ����� �������������.
    """

    # �������� ������ ���� ����������� � ����������
    files = [f for f in glob.glob(path + "**/*.png")]

    # ���������� ��� ������� ������
    H, W = 0, 0
    projections = {}

    # ������ �� ���� ������������

    print("��������� �������� � �����������")
    for file in files:
        # �������� �������� ����� ���������� �� ��������
        # �� �� �������� ����� ��������
        serial = get_serial(file)

        # �������� �������� �� �����������
        projection, minX, minY, maxX, maxY = get_projection(file, scale)

        projections[serial] = projection

        H, W = projection.shape
        # if maxY - minY > H:
        #     H = maxY - minY
        #
        # if maxX - minX > W:
        #     W = maxX - minX

    # ������� ������ ����� �� ������������ ��������
    print("�������� ���� ������ �����")
    point_cloud = get_point_cloud_cube(H, W)

    temp = 3
    # ������� ��� ������� ����������� ��������� ��������
    for angle in projections:
        print("��������� �������� ���� " + str(angle))
        model = make_model_from_mask(projections[angle], angle)
        print(point_cloud.shape)
        print(model.shape)
        point_cloud = np.multiply(point_cloud, model)

    print("���������� ���������� XYZ")
    xyz_path = 'point_cloud.xyz'
    save_xyz(point_cloud, xyz_path)
    pc = pv.PolyData(np.genfromtxt(xyz_path, delimiter=' ', dtype=np.float32))
    pc.plot()

    # # ������� ������ ����� �� ������������ ��������
    # print("�������� ���� ������ �����")
    # point_cloud = get_point_cloud_cube(H, W)
    #
    # # �������� �������� ������� �� ������ �����
    # for angle in projections:
    #     print("��������� �������� ���� " + str(angle))
    #     cut_shape(point_cloud, angle, projections[angle])

    # ������� ������������ �������

    # ��������� ��������� � ������� XYZ
    # print("���������� ���������� XYZ")
    # xyz_path = 'point_cloud.xyz'
    # save_xyz(point_cloud, xyz_path)

    # ��������������� ���������

    # ������� �����������

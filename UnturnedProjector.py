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
    �������� ����� �� ���� (��������) �����������.
    :param image: ���� � �����
    :return: �������� �����
    """
    # ������� ���������� ����� �� ����� �����
    serial = int(re.search(r"\d{3}", os.path.basename(image)).group())

    return serial


def get_projection(path, scale=100, intensity=50):
    """
    �������� �������� ������� �� ���������� � ������� ������ ��������� ����� � ������ ������ (xl, xr, y)
    � ������� ������� ��������� � ����� ������� ���� �����������
    :param path: ���� � �����������
    :param scale: ��������������� ����������� � ��������� ��� ��������� ��������, �� ������� ������ 100
    :return: ������ �������� (xl, xr, y), ������ � ������, � ����� ���������� ������� ����� ����� ��������
    """
    # ������ ����������� �� �����
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # ������ ������ ����������� ��� ����, ����� ��������� ��������, �� �������� ����������
    # ����� ���, ��� ��������� ��� � ������������ ����������� ������
    if scale < 100:
        image = cv2.resize(image,
                           (int(image.shape[1] * scale / 100),
                            int(image.shape[0] * scale / 100)))

    # ������ ������ ��� �������� ��������� (�� numpy)
    lines = []

    # ������� ������� ������
    minY, minX = image.shape
    maxY, maxX = 0, 0

    # ��������� ������� ������� ������� � ���������� � ��������
    # ����� ��������� ������� ������ �� ���� ��� ������������
    # �������� �� ������� �����������
    for i in range(image.shape[0]):
        # ���� �������� intensity ������� ������ 50
        row = np.where(image[i, :] > intensity)
        # ���� � ������ ������ ������ �������
        if len(row[0]) >= 2:

            # ��������� ������� �� Y
            if minY == image.shape[0]:
                minY = i
            else:
                maxY = i

            # ��������� ������� �� X
            if row[0][0] < minX:
                minX = row[0][0]
            if row[0][-1] > maxX:
                maxX = row[0][-1]

            # ������� � ������� �������� ����� ��� ������ [xl, xr, y] � ������� float
            lines.append([float(row[0][0]), float(row[0][-1]), float(i)])

    # ��������� ������� ������ �� ��������
    height = maxY - minY
    width = maxX - minX

    # �������� ������ ������ � numpy ��� ��������� ����������� ����������
    np_lines = np.array(lines)

    # cv2.imshow("image", image)
    # cv2.waitKey()

    # print("�������: {}, {}, {}, {}".format(minX, minY, height, width))

    return np_lines, height, width, minX, minY


def get_projections_from_files(path, scale):
    """
    �������� ������� � ����������
    :param path: ���� � ���������� � �������������
    :param scale: ������� ����������� � ���������
    :return: ���������� ������� ��������, � ����� ������������ ������� ������ � ���������� ������ ������ ��� ������������
    """
    # �������� ������ ���� ����������� � ����������
    files = [f for f in glob.glob(path + "**/*.png", recursive=True)]

    # ������� ��� ��������
    projections = {}

    # ������������ ������� ������
    max_width, max_height = 0, 0

    # ���������� ����� ������� ����� ������ ��� ������������
    norm_X, norm_Y = 1000, 1000

    # ������ �������� ������ �������� � ��������� ����������� ��������� ������ ������
    print("��������� �������� � �����������")
    for file in files:
        # �������� �������� ����� ���������� �� ��������
        # �� �� �������� ����� ��������
        serial = get_serial(file)

        # �������� �������� �� ����������� � ������� ������ ����� ������
        # ����� ���������� �������� ������������ ������� ����� �� ���������� � ��������
        projection, height, width, x0, y0 = get_projection(file, scale, intensity=35)

        # ��������� ������������ ������� ���� ������
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height

        # ��������� ����� ������� ����� ��� ����������� ������������
        if x0 < norm_X:
            norm_X = x0
        if y0 < norm_Y:
            norm_Y = y0

        # ��������� �������� � ������� � ����� � �������� �����
        projections[serial] = np.array(projection)

    # ����� ��������� ����� ������� ��������� � ����� �����������
    norm_X += round(max_width / 2)
    norm_Y += round(max_height / 2)

    # print("������� ������: {}, {}, {}, {}".format(max_width, max_height, norm_X, norm_Y))

    return projections, max_width, max_height, norm_X, norm_Y


def make_point_cloud_cube(max_height, max_width):
    """
    ������ ������ ����� ��� ������������ ������� � ������� ������� x, y, z � ������� ������� ��������� � ������ ����
    :param max_height: ������������ ������ ������
    :param max_width: ������������ ������ ������
    :return: ������ ����� � ����� ���� � ������� [x, y, z]
    """
    print("�������� ������ ����� {} x {} x {}".format(max_width, max_width, max_height))

    h_range = np.arange(-max_height / 2, max_height / 2)
    w_range = np.arange(- max_width / 2, max_width / 2)

    point_cloud = np.array(np.meshgrid(w_range, w_range, h_range)).T.reshape(-1, 3)

    return point_cloud


def normalize(projection, nx, ny):
    """
    ���������� ��������������� ��������
    :param projection: �������� �� �����������
    :param nx: ����������� ������������ �
    :param ny: ����������� ������������ Y
    :return: ��������������� ��������
    """

    # ������������ ������� ������ ��� ������������ ���� �������� ��� ���� ������
    # �������� � ����������� �������� ������� ����� �������� � ������������ ��������, �� ���������� ������
    # �������� �� ���� �������� ���������� ������ �����, ����� ������ �������� ��������������� ������� ���������
    projection -= [nx, nx, ny]

    # ��������� � �������
    # ����� ������������, ������ ��� ������������ ������� ����� numpy (�� ��� � ����� �� �����, ��� ��� �������� ���������)
    lines = {}
    for line in projection:
        lines[round(line[2])] = [line[0], line[1]]

    return lines

    return projection


def turn_point_cloud_horizontally(point_cloud, angle):
    """
    ������� ������ ����� �� ����������� (������������ ��� Z) �� ����������� ����
    :param point_cloud: ������ �����, ������� ��������� ���������
    :param angle: ����, �� ������� ��������� ��������� ������ �����
    :return: ������������������ ������ �����
    """
    # ���� �������� � ��������
    radian_angle = angle * np.pi / 180.0

    # ����� ��������� ����� � �������
    cos = np.cos(radian_angle)
    sin = np.sin(radian_angle)

    # �������� ������� � x � y ������������
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]

    # �������� ���������� ��� �������� �� ������ ����
    point_cloud[:, 0] = x * cos - y * sin
    point_cloud[:, 1] = x * sin + y * cos

    return point_cloud


def cut_into_point_cloud(point_cloud, lines, angle):
    # ������ ������ ��� ����������
    result = []

    # ������� �������� � �������
    alpha = (angle / 180.0) * np.pi

    # ��������� ������� ���� ��� ����� �������, ���������� ������������������ �� �������
    # ������ ������� ���� ������ ����� � ���������� ��������� ������
    indices = np.unique(point_cloud[:, 2], return_index=True)[1][1:]

    # �������� �� ���� �������� ������ �� ���������� ����� �� ����� ������
    # (������� �������, ��� ������ ��� �������� ����� �� �������)
    sliced_array = np.split(point_cloud, indices)

    # ������ �� ������� ��������������� ������
    for sl in sliced_array:
        try:
            # ������� �� ������� ���������� ��������� �������� ��� ��������������� ������
            xl, xr = lines[round(sl[0][2])]

            # TODO: �������� ���������� ����� �� ��������
            # xp = np.sqrt(sl[:, 0] ** 2 + sl[:, 1] ** 2) * np.cos(alpha + np.arctan2(sl[:, 0], sl[:, 1]))

            # TODO: ������� ������ ����������� ��� ��������� �������� �����
            sl = sl[
                np.where(np.sqrt(sl[:, 0] ** 2 + sl[:, 1] ** 2) * np.cos(alpha + np.arctan2(sl[:, 1], sl[:, 0])) > xl)]
            sl = sl[
                np.where(np.sqrt(sl[:, 0] ** 2 + sl[:, 1] ** 2) * np.cos(alpha + np.arctan2(sl[:, 1], sl[:, 0])) < xr)]

            # ������� �� � ������ �������
            result.append(sl)

        # �� ������ ���������� � ������� �������� ��� ������ ������ ���������� ���
        except KeyError or IndexError:
            continue

    # ���������� ���������� ������
    return np.concatenate(result)


def viscera_disposal(point_cloud):
    print("��������� �������������")
    # ������ ������ ����� ���������� ������������ �� z, ����������� ���������

    # ������ ������ ��� ����������
    result = []

    # ������ ������� ���� ������ ����� � ���������� ��������� ������
    xy_indices = np.unique(point_cloud[:, 2], return_index=True)[1][1:]

    # �������� �� ���� �������� ������ �� ���������� ����� �� ����� ������
    # (������� �������, ��� ������ ��� �������� ����� �� �������)
    sliced_array = np.split(point_cloud, xy_indices)

    # ������ �� ������� ��������������� ������
    for sl in sliced_array:

        sl = sl[sl[:, 0].argsort()]

        # ������� ����� ������ �� �
        x_id = np.unique(sl[:, 0], return_index=True)[1][1:]

        # ������� ������ ������
        rows = np.split(sl, x_id)

        # ����� ����������� � ������������ ����� � ������ � ������� �� � ����������
        for row in rows:
            try:
                minX = row[np.argmin(row[:, 1], axis=0)]
                result.append(minX)

                maxX = row[np.argmax(row[:, 1], axis=0)]
                result.append(maxX)
            except ValueError:
                continue

        sl = sl[sl[:, 1].argsort()]

        # ������ �������� ������ �� Y
        y_id = np.unique(sl[:, 1], return_index=True)[1][1:]

        # ������� ������� ������
        cols = np.split(sl, y_id)

        # ����� ����������� � ������������ ����� � ������� � ������� �� � ����������
        for col in cols:
            try:
                minY = col[np.argmin(col[:, 0], axis=0)]
                result.append(minY)

                maxY = col[np.argmax(col[:, 0], axis=0)]
                result.append(maxY)
            except ValueError:
                continue

    # ���������� ������������� ������ ����� �� ��� X ��� ��������� ������� YZ
    point_cloud = point_cloud[point_cloud[:, 0].argsort()]

    # ������ ������� ���� ������ ����� � ���������� ��������� �� ��� X
    yz_indices = np.unique(point_cloud[:, 0], return_index=True)[1][1:]

    # �������� �� ��� �������������� ��������������� ������ �� ���������� ����� � ����� ����������� �
    # ���������� ������� YZ ������ �� ������ �����
    sliced_array = np.split(point_cloud, yz_indices)

    # ������ �� ������������ ������� YZ
    for sl in sliced_array:

        sl = sl[sl[:, 1].argsort()]

        # ������� ������� ������ �� Z
        y_id = np.unique(sl[:, 1], return_index=True)[1][1:]

        # ������� ������ ������
        pils = np.split(sl, y_id)

        # ����� ����������� � ������������ ����� � ������ � ������� �� � ����������
        for pil in pils:
            try:
                minZ = pil[np.argmin(pil[:, 2], axis=0)]
                result.append(minZ)

                maxZ = pil[np.argmax(pil[:, 2], axis=0)]
                result.append(maxZ)
            except ValueError:
                continue

    # ���������� ���������� ������
    return np.vstack(result)


def save_xyz(point_cloud, path):
    """
    ��������� ������ ����� � ������� XYZ
    :param point_cloud: ������ �����
    :param path: ���� ��� ���������� ����� XYZ
    """

    tic = time.perf_counter()

    print("���������� ���������� XYZ")

    with open(path, 'w+') as file_object:
        for point in point_cloud:
            file_object.write(str(point[0]) + ' ' +
                              str(point[1]) + ' ' +
                              str(point[2]) + '\n')

    toc = time.perf_counter()
    print(round(toc - tic, 3))


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

    # ��� ������� �� ���������� ����������� ������, ���������� ����� ��������
    # ������ ������� ������� �� ��� �����
    for facet in cube:
        x, y, z = point
        voxel_string += get_facet(x, y, z, facet)

    return voxel_string


def generate_voxels(point_cloud, path):
    tic = time.perf_counter()
    print("������������")

    list_cloud = list(point_cloud)

    with open(path, 'w+', buffering=1024) as file_object:
        file_object.write("solid model\n")

        for point in list_cloud:
            file_object.write(get_voxel(point))

        file_object.write("endsolid model")

    toc = time.perf_counter()
    print(round(toc - tic, 3))


def generate_surface(point_cloud, neighbours, path):
    tic = time.perf_counter()

    print("��������� ����������� � ����������� �������� ����� " + str(neighbours))
    mesh = pv.PolyData(point_cloud).reconstruct_surface(nbr_sz=neighbours)
    fixed = pmf.MeshFix(mesh)
    fixed.repair()
    fixed.mesh.save(path)

    toc = time.perf_counter()
    print(round(toc - tic, 3))

    # fixed.mesh.plot()


def generate_model(path, scale, neighbours=15):
    """
    �������� �����, ������������ ������ � �����������
    :param path: ���� � ���������� � �������������
    :param scale: ����������� ������ ����������� � ���������
    :param radial: ���� �������� ������ �� ������������
    :return: ����� ������ �� ����������, �� ��������� ����� ������� � ���������� � �������������
    """
    g_tic = time.perf_counter()  # ���������� ������� �������

    # ������� �������� � �����������, � ����� ������������ ������� ������ � ����� ������� ����� ��� ������������
    # � ������� �������� ����� ������� ��������� � ����� ������� ����� ������������� �����������
    projections, max_width, max_height, norm_X, norm_Y = get_projections_from_files(path, scale)

    # ������� ������ ����� ��� ������������ ������� � ������� ������� x, y, z
    # ����� ������� ��������� ������ ����� ������ ���� � ������ ����
    tic = time.perf_counter()
    point_cloud = make_point_cloud_cube(max_height, max_width)
    toc = time.perf_counter()
    print(round(toc - tic, 3))

    # ������ �������� ����������� ��� �������� � �������� �������� ������
    for angle in projections:
        print("��������� ����� " + str(angle) + " �� ������ �����")

        tic = time.perf_counter()

        # ������������ ��������, ���������� ������� ��������������� �����
        # ����� ������������ ��� �������� ����� ����� ������� ��������� � ������ ������������� ������
        lines = normalize(projections[angle], norm_X, norm_Y)

        # �������������� ������� ������
        # ������ �������� ��������� ��������� ��� ��� ���������
        # if angle != 0:
        #     point_cloud = turn_point_cloud_horizontally(point_cloud, radial)

        # ������� �� ������� ��� �����, �� ������������� ��������
        point_cloud = cut_into_point_cloud(point_cloud, lines, angle)

        toc = time.perf_counter()
        print(round(toc - tic, 3))

    # ������� ��������� ���������
    # pv.plot(pv.PolyData(point_cloud))

    # ������� ���������
    # pv.plot(pv.PolyData(point_cloud))

    # ������� ������������ ������ �����
    # ������ �������� ������������� ����� ����� ������ ����� � ������ �� ������, ��� ����� ���� ������ � ������ �����
    tic = time.perf_counter()
    point_cloud = viscera_disposal(point_cloud)
    toc = time.perf_counter()
    print(round(toc - tic, 3))

    # ����� ������ ������� ���������� ��� ������
    print("����� ����� ����������")
    g_toc = time.perf_counter()
    print(round(g_toc - g_tic, 3))

    # ������� ���������
    pc = pv.PolyData(point_cloud)
    # pv.plot(pc)

    # ��������� ������ �����
    save_xyz(point_cloud, path + '/{}/point_cloud_{}.xyz'.format(scale, scale))

    # ��������������� ������ �����
    # list_cloud = list(point_cloud)
    generate_voxels(point_cloud, path + '/{}/voxels_{}.stl'.format(scale, scale))

    # ������� ����������� ������ �����
    generate_surface(pc, neighbours, path + '/{}/mesh_{}.stl'.format(scale, scale))

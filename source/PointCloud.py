# -*- coding: cp1251 -*-
import numpy as np
import cupy as cp
import pyvista
import time

from numba import cuda


class PointCloud:

    def __init__(self):
        self.point_cloud = None

    def make_point_cloud_cube(self, width, height):
        tic = time.perf_counter()

        height_range = np.arange(round(-height / 2), round(height / 2))
        width_range = np.arange(round(-width / 2), round(width / 2))

        self.point_cloud = np.array(np.meshgrid(width_range, width_range, height_range)).T.reshape(-1, 3)

        toc = time.perf_counter()
        diff = round(toc - tic, 3)
        print("PC gen:  " + str(diff))

    def cut_into_point_cloud(self, contour, angle):
        # ������ ������ ��� ����������
        result = []

        # ������� �������� � �������
        alpha = -(angle / 180.0) * np.pi

        # ��������� ������� ���� ��� ����� �������, ���������� ������������������ �� �������
        # ������ ������� ���� ������ ����� � ���������� ��������� ������
        indices = np.unique(self.point_cloud[:, 2], return_index=True)[1][1:]

        # indices_tuple = tuple(np.atleast_1d(indices))
        # for e in indices_tuple:
        #     e = int(e)

        # �������� �� ���� �������� ������ �� ���������� ����� �� ����� ������
        # (������� �������, ��� ������ ��� �������� ����� �� �������)
        sliced_array = np.split(self.point_cloud, indices)

        # ������ �� ������� ��������������� ������
        for sl in sliced_array:
            sl = cp.array(sl)
            try:
                # ������� �� ������� ���������� ��������� �������� ��� ��������������� ������
                xl, xr = contour[round(sl[0][2])]

                # ������� ������ ����������� ��� ��������� �������� �����
                sl = sl[
                    np.where(
                        np.sqrt(sl[:, 0] ** 2 + sl[:, 1] ** 2) * np.cos(alpha + np.arctan2(sl[:, 1], sl[:, 0])) > xl)]
                sl = sl[
                    np.where(
                        np.sqrt(sl[:, 0] ** 2 + sl[:, 1] ** 2) * np.cos(alpha + np.arctan2(sl[:, 1], sl[:, 0])) < xr)]

                # ������� �� � ������ �������
                result.append(sl)

            # �� ������ ���������� � ������� �������� ��� ������ ������ ���������� ���
            except KeyError or IndexError:
                continue

        # pc = pyvista.PolyData(np.concatenate(result))
        # pc.plot()

        # ���������� ���������� ������
        self.point_cloud = np.concatenate(result)

    def viscera_disposal(self):
        # ������ ������ ����� ���������� ������������ �� z, ����������� ���������

        # ������ ������ ��� ����������
        result = []

        # ������ ������� ���� ������ ����� � ���������� ��������� ������
        xy_indices = np.unique(self.point_cloud[:, 2], return_index=True)[1][1:]

        # �������� �� ���� �������� ������ �� ���������� ����� �� ����� ������
        # (������� �������, ��� ������ ��� �������� ����� �� �������)
        sliced_array = np.split(self.point_cloud, xy_indices)

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
        self.point_cloud = self.point_cloud[self.point_cloud[:, 0].argsort()]

        # ������ ������� ���� ������ ����� � ���������� ��������� �� ��� X
        yz_indices = np.unique(self.point_cloud[:, 0], return_index=True)[1][1:]

        # �������� �� ��� �������������� ��������������� ������ �� ���������� ����� � ����� ����������� �
        # ���������� ������� YZ ������ �� ������ �����
        sliced_array = np.split(self.point_cloud, yz_indices)

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
        self.point_cloud = np.vstack(result)

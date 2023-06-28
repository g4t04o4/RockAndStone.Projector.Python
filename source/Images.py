# -*- coding: cp1251 -*-
import re
import os
import glob
import numpy as np
import cv2


class Images:
    def __init__(self, path, scale):

        # ���� � ���������� � �������������
        self.path = path

        # ������� �����������
        self.scale = scale

        # ������� ��� ��������
        self.masks = {}

        # ��������� ��������
        self.intensity = 25

        # ���������� ������� ������ �� �����������
        self.center_dict = {}

        # ����� ����� � ������ ��������� ��������
        self.left_right_masks = {}

        # ����� ������� � ������ ��������� ��������
        self.up_down_masks = {}

        # ������������ ������� ������
        self.x1, self.y1 = 0, 0

        # ���������� ����� ������� ����� ������ ��� ������������
        self.x0, self.y0 = 1000, 1000

    def load_gray_image(self, image_path):
        # ������ ����������� �� �����
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # ������ ������ ����������� ��� ����, ����� ��������� ��������, �� �������� ����������
        # ����� ���, ��� ��������� ��� � ������������ ����������� ������
        if self.scale < 100:
            image = cv2.resize(image,
                               (int(image.shape[1] * self.scale / 100),
                                int(image.shape[0] * self.scale / 100)))
        return image

    def get_contour(self, image):
        # ������ ������ ������ ����������� ��� ������
        output = np.zeros(image.shape)

        # ������� ����� �� ���������� ��������
        ret, thresh = cv2.threshold(image, self.intensity, 255, 0)

        # ����� � ����� �������
        cnt, hier = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # ������� ���������� ������
        c = max(cnt, key=cv2.contourArea)

        # �������� ���������� ������ �� ������ �����������
        cv2.drawContours(output, c, -1, color=255, thickness=cv2.FILLED)

        return output

    def get_single_mask(self, file_path, angle):
        # ��������� ����������� � ��������� � �����-����� ������
        image = self.load_gray_image(file_path)

        # ������� ������ � �����������
        contour = self.get_contour(image)

        # ������� ������� ������
        minY, minX = image.shape
        maxY, maxX = 0, 0

        # TODO: ����� �������� ����������������� �������������� ������ � ��������
        # �������� �� ������� �����������

        left_right = []

        y = contour.shape[0]
        for i in range(y):
            # ���� �� ������ �� �������, �������� ������� ���� ����������
            row = np.where(contour[i, :] > 0)

            # ���� � ������ ������ ������ �������
            if len(row[0]) >= 2:

                # ��������� ������� �� Y
                if minY == y:
                    minY = i
                else:
                    maxY = i

                # ��������� ������� �� X
                if row[0][0] < minX:
                    minX = row[0][0]
                if row[0][-1] > maxX:
                    maxX = row[0][-1]

                # ������� � ������� ����� ��� ������ [xl, xr, y] � ������� float
                left_right.append([float(row[0][0]), float(row[0][-1]), float(i)])

        #  ��������� ������������ ������� ���� ������
        if maxX > self.x1:
            self.x1 = maxX
        if maxY > self.y1:
            self.y1 = maxY

        # ��������� ����� ������� ����� ��� ����������� ������������
        if minX < self.x0:
            self.x0 = minX
        if minY < self.y0:
            self.y0 = minY

        # TODO: ������� ����������������� ������������ ������
        # �������� �� �������� �����������

        up_down = []

        x = contour.shape[1]
        for i in range(x):
            # ���� �� ������� �� �������, �������� ������� ���� ����������
            col = np.squeeze(np.array(np.where(contour[:, i] > 0)))

            try:
                if len(col) >= 2:
                    # ������� � ������� ����� ��� ������ [x, yu, yd]
                    up_down.append([float(i), float(col[0]), float(col[-1])])
            except TypeError:
                continue
        # ��������� ������� ������� ������� � ���������� � ��������
        # ����� ��������� ������� ������ �� ���� ��� ������������

        # �������� ����� ��� ������� ������
        contour = contour[minY - 1:maxY + 1, minX - 1:maxX + 1]

        # ���������� ��������� ���������� ������ ������ ������������ ������ �������� ���� ������������� ���� ��� ������������
        self.center_dict[angle] = [round((minX + maxX) / 2), round((minY + maxY) / 2)]

        # cv2.imshow("mask", contour)
        # cv2.waitKey()

        # ��������� �������� � ������� � ����� � �������� �����
        self.masks[angle] = np.array(contour)
        self.up_down_masks[angle] = np.array(up_down)
        self.left_right_masks[angle] = np.array(left_right)

    def normalize(self):
        # TODO: ���������� ������������� �������

        for angle in self.center_dict:
            self.up_down_masks[angle] -= [self.center_dict[angle][0],
                                          self.center_dict[angle][1],
                                          self.center_dict[angle][1]]

            self.left_right_masks[angle] -= [self.center_dict[angle][0] - 1,
                                             self.center_dict[angle][0] + 1,
                                             self.center_dict[angle][1]]

    def generate_masks(self):
        # �������� ������ ���� ����������� � ����������
        files = [f for f in glob.glob(self.path + "**/*.png", recursive=True)]

        # ������ �������� ������ �������� � ��������� ����������� ��������� ������ ������
        for file_path in files:
            # �������� �������� ����� ���������� �� ��������
            # �� �� �������� ����� ��������
            angle = int(re.search(r"\d{3}", os.path.basename(file_path)).group())

            print("������������ ����������� ���� " + str(angle))

            # �������� �������� �� ����������� � ������� ������ ����� ������
            # ����� ���������� �������� ������������ ������� ����� �� ���������� � ��������
            self.get_single_mask(file_path, angle)

        self.normalize()

        # cv2.imshow("mask", mask)
        # cv2.waitKey()

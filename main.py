# -*- coding: cp1251 -*-

import Projector as pj

dir_path = "C:/Images/quartz_10"

projector = pj.Projector(dir_path, scale=100)

projector.generate_model()

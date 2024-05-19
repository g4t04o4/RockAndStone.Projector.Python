# -*- coding: cp1251 -*-
import numpy as np
import Projector as pj

image_path = "C:/Images/dfive/crops"
angled_path = "C:/Images/dfive/crops"
damage_path = "C:/Images/dfive/masks"

scale = 25
angle = 10

projector = pj.Projector(image_path, angled_path, damage_path, scale, angle)

projector.generate_model()

# -*- coding: cp1251 -*-
# import resource

import UnturnedProjector as upj

# def plot_file(path):
#     mesh = pv.read(path)
#     mesh.plot()


dir_path = "C:/Images/quartz_10"

upj.generate_model("C:/Images/quartz_10", scale=25, neighbours=15)

# plot_file("mesh.stl")

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.utils.data import DataLoader, Dataset

from sympy import Symbol, Eq, Abs, tanh

import numpy as np

from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.geometry import Parameterization, Parameter
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node


# define sympy varaibles to parametize domain curves
x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
x_pos = Symbol("x_pos")

# parametric variation
fin_height_s = Symbol("fin_height_s")
fin_length_s = Symbol("fin_length_s")
fin_thickness_s = Symbol("fin_thickness_s")
height_s_range = (0.025, 0.030)
length_s_range = (0.180, 0.220)
thickness_s_range = (0.001, 0.003)
param_ranges = {
    fin_height_s: height_s_range,
    fin_length_s: length_s_range,
    fin_thickness_s: thickness_s_range,
}
fixed_param_ranges = {
    fin_height_s: 0.027,
    fin_length_s: 0.195,
    fin_thickness_s: 0.0025,
}

# geometry params for domain
channel_origin = (-0.500, -0.030, -0.250)
channel_dim = (1.0, 0.060, 0.500)
heat_sink_base_origin = (-0.195, -0.030, -0.102)
heat_sink_base_dim = (0.195, 0.008, 0.2049)
fin_origin = (heat_sink_base_origin[0] + 0.0975 - fin_length_s / 2, 
              heat_sink_base_origin[1] + heat_sink_base_dim[1], 
              -0.10245)
fin_dim = (fin_length_s, fin_height_s, fin_thickness_s)  # two side fins
total_fins = 24  # total fins
flow_box_origin = (-0.240, -0.030, -0.250)
flow_box_dim = (0.300, 0.060, 0.500)
source_origin = (-0.120, -0.030, 0.0075)
source_dim = (0.050, 0.0, 0.040)
source_area = 0.002


# define geometry
class ThreeFin(object):
    def __init__(self, parameterized: bool = False):
        # set param ranges
        if parameterized:
            pr = Parameterization(param_ranges)
            self.pr = param_ranges
        else:
            pr = Parameterization(fixed_param_ranges)
            self.pr = fixed_param_ranges

        # channel
        self.channel = Channel(
            channel_origin,
            (
                channel_origin[0] + channel_dim[0],
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            parameterization=pr,
        )

        # three fin heat sink
        heat_sink_base = Box(
            heat_sink_base_origin,
            (
                heat_sink_base_origin[0] + heat_sink_base_dim[0],  # base of heat sink
                heat_sink_base_origin[1] + heat_sink_base_dim[1],
                heat_sink_base_origin[2] + heat_sink_base_dim[2],
            ),
            parameterization=pr,
        )

        fin = Box(
            fin_origin,
            (
                fin_origin[0] + fin_dim[0],
                fin_origin[1] + fin_dim[1],
                fin_origin[2] + fin_dim[2],
            ),
            parameterization=pr,
        )
        
        # 23 fins
        fin_center = (
            fin_origin[0] + fin_dim[0] / 2,
            fin_origin[1] + fin_dim[1] / 2,
            fin_origin[2] + fin_dim[2] / 2,
        )
        gap = (heat_sink_base_dim[2] - fin_dim[2]) / (total_fins - 1)  # gap between fins
        '''fin = fin.repeat(
            gap,
            repeat_lower=(0, 0, 0),
            repeat_higher=(0, 0, total_fins - 1),
            center=fin_center,
        )
        self.three_fin = heat_sink_base + fin'''
        
        fins = fin
        for i in range(1, total_fins):
            fins = fins + fin.translate([0, 0, i * gap])

        self.three_fin = heat_sink_base + fins
        # entire geometry
        self.geo = self.channel - self.three_fin

        # low and high resultion geo away and near the heat sink
        flow_box = Box(
            flow_box_origin,
            (
                flow_box_origin[0] + flow_box_dim[0],  # base of heat sink
                flow_box_origin[1] + flow_box_dim[1],
                flow_box_origin[2] + flow_box_dim[2],
            ),
        )
        self.lr_geo = self.geo - flow_box
        self.hr_geo = self.geo & flow_box
        lr_bounds_x = (channel_origin[0], channel_origin[0] + channel_dim[0])
        lr_bounds_y = (channel_origin[1], channel_origin[1] + channel_dim[1])
        lr_bounds_z = (channel_origin[2], channel_origin[2] + channel_dim[2])
        self.lr_bounds = {x: lr_bounds_x, y: lr_bounds_y, z: lr_bounds_z}
        hr_bounds_x = (flow_box_origin[0], flow_box_origin[0] + flow_box_dim[0])
        hr_bounds_y = (flow_box_origin[1], flow_box_origin[1] + flow_box_dim[1])
        hr_bounds_z = (flow_box_origin[2], flow_box_origin[2] + flow_box_dim[2])
        self.hr_bounds = {x: hr_bounds_x, y: hr_bounds_y, z: hr_bounds_z}

        # inlet and outlet
        self.inlet = Plane(
            channel_origin,
            (
                channel_origin[0],
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            -1,
            parameterization=pr,
        )
        self.outlet = Plane(
            (channel_origin[0] + channel_dim[0], channel_origin[1], channel_origin[2]),
            (
                channel_origin[0] + channel_dim[0],
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            1,
            parameterization=pr,
        )

        # planes for integral continuity
        self.integral_plane = Plane(
            (x_pos, channel_origin[1], channel_origin[2]),
            (
                x_pos,
                channel_origin[1] + channel_dim[1],
                channel_origin[2] + channel_dim[2],
            ),
            1,
        )

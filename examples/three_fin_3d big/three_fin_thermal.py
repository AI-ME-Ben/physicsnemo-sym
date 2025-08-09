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

import os
import warnings

import torch
from sympy import Symbol, Eq, Abs, tanh, Or, And
import itertools
import numpy as np

import physicsnemo.sym
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes, Curl
from physicsnemo.sym.eq.pdes.basic import NormalDotVec, GradNormal
from physicsnemo.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion

from three_fin_geometry import *

from physicsnemo.sym.distributed.manager import DistributedManager

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(message)s',
    datefmt='%H:%M:%S'
)
@physicsnemo.sym.main(config_path="conf", config_name="conf_thermal")
def run(cfg: PhysicsNeMoConfig) -> None:
    # fluid params
    nu = 0.02
    rho = 1
    # heat params
    k_fluid = 0.024
    k_solid = 205.0
    D_solid = 0.10
    D_fluid = 0.02
    source_grad = 1.5
    # make thermal equations
    ad = AdvectionDiffusion(T="theta_f", rho=rho, D=D_fluid, dim=3, time=False)
    dif = Diffusion(T="theta_s", D=D_solid, dim=3, time=False)
    dif_inteface = DiffusionInterface("theta_f", "theta_s", k_fluid, k_solid, dim=3, time=False)
    f_grad = GradNormal("theta_f", dim=3, time=False)
    s_grad = GradNormal("theta_s", dim=3, time=False)

    # make network arch
    input_keys = [
            Key("x"),
            Key("y"),
            Key("z"),
            Key("fin_height_s"),
            Key("fin_length_s"),
            Key("fin_thickness_s"),
        ]
    c = Curl(("a", "b", "c"), ("u", "v", "w"))
    equation_nodes += c.make_node()
    output_keys = [Key("a"), Key("b"), Key("c"), Key("p")]

    flow_net = ModifiedFourierNetArch(
        input_keys=input_keys,
        output_keys=output_keys,
        adaptive_activations=cfg.custom.adaptive_activations,
    )
    thermal_f_net = ModifiedFourierNetArch(
        input_keys=input_keys,
        output_keys=[Key("theta_f")],
        adaptive_activations=cfg.custom.adaptive_activations,
    )
    thermal_s_net = ModifiedFourierNetArch(
        input_keys=input_keys,
        output_keys=[Key("theta_s")],
        adaptive_activations=cfg.custom.adaptive_activations,
    )


    # make list of nodes to unroll graph on
    logging.info("開始建立網路節點")
    thermal_nodes = (
        ad.make_nodes()
        + dif.make_nodes()
        + dif_inteface.make_nodes()
        + f_grad.make_nodes()
        + s_grad.make_nodes()
        + [flow_net.make_node(name="flow_network", optimize=False)]
        + [thermal_f_net.make_node(name="thermal_f_network")]
        + [thermal_s_net.make_node(name="thermal_s_network")]
    )

    geo = ThreeFin(parameterized=cfg.custom.parameterized)
    logging.info("Geo 建立完成")

    # params for simulation
    # heat params
    inlet_t = 293.15 / 273.15 - 1.0
    grad_t = 360 / 273.15

    # make flow domain
    logging.info("建立 flow domain")
    thermal_domain = Domain()

    # inlet
    logging.info("inlet建立")
    constraint_inlet = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.inlet,
        outvar={"theta_f": inlet_t},
        batch_size=cfg.batch_size.Inlet,
        criteria=Eq(x, channel_origin[0]),
        lambda_weighting={"theta_f": 1.0},  # weight zero on edges
        parameterization=geo.pr,
        quasirandom=cfg.custom.quasirandom,
    )
    thermal_domain.add_constraint(constraint_inlet, "inlet")

    # outlet
    logging.info("outlet建立")
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.outlet,
        outvar={"normal_gradient_theta_f": 0},
        batch_size=cfg.batch_size.Outlet,
        criteria=Eq(x, channel_origin[0] + channel_dim[0]),
        lambda_weighting={"normal_gradient_theta_f": 1.0},  # weight zero on edges
        parameterization=geo.pr,
        quasirandom=cfg.custom.quasirandom,
    )
    thermal_domain.add_constraint(constraint_outlet, "outlet")

    # channel walls insulating
    logging.info("wall_criteria建立")
    def wall_criteria(invar, params):
        sdf = geo.three_fin.sdf(invar, params)
        return np.less(sdf["sdf"], -1e-5)

    channel_walls = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.channel,
        outvar={"normal_gradient_theta_f": 0},
        batch_size=cfg.batch_size.ChannelWalls,
        criteria=wall_criteria,
        lambda_weighting={"normal_gradient_theta_f": 1.0},
        parameterization=geo.pr,
        quasirandom=cfg.custom.quasirandom,
    )
    thermal_domain.add_constraint(channel_walls, "channel_walls")

    # fluid solid interface
    logging.info("interface_criteria 建立")
    def interface_criteria(invar, params):
        sdf = geo.channel.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    fluid_solid_interface = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.three_fin,
        outvar={
            "diffusion_interface_dirichlet_theta_f_theta_s": 0,
            "diffusion_interface_neumann_theta_f_theta_s": 0,
        },
        batch_size=cfg.batch_size.SolidInterface,
        criteria=interface_criteria,
        parameterization=geo.pr,
        quasirandom=cfg.custom.quasirandom,
    )
    thermal_domain.add_constraint(fluid_solid_interface, "fluid_solid_interface")

    # heat source
    logging.info("heat source 建立")
    sharpen_tanh = 60.0
    source_func_xl = (tanh(sharpen_tanh * (x - source_origin[0])) + 1.0) / 2.0
    source_func_xh = (
        tanh(sharpen_tanh * ((source_origin[0] + source_dim[0]) - x)) + 1.0
    ) / 2.0
    source_func_zl = (tanh(sharpen_tanh * (z - source_origin[2])) + 1.0) / 2.0
    source_func_zh = (
        tanh(sharpen_tanh * ((source_origin[2] + source_dim[2]) - z)) + 1.0
    ) / 2.0
    gradient_normal = (
        grad_t * source_func_xl * source_func_xh * source_func_zl * source_func_zh
    )
    heat_source = PointwiseBoundaryConstraint(
        nodes=thermal_nodes,
        geometry=geo.three_fin,
        outvar={"normal_gradient_theta_s": gradient_normal},
        batch_size=cfg.batch_size.HeatSource,
        criteria=Eq(y, source_origin[1]),
        quasirandom=cfg.custom.quasirandom,
    )
    thermal_domain.add_constraint(heat_source, "heat_source")

    # flow interior low res away from three fin
    logging.info("lr_flow_interior 建立")
    lr_flow_interior = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=geo.geo,
        outvar={"advection_diffusion_theta_f": 0},
        batch_size=cfg.batch_size.InteriorLR,
        criteria=Or(x < flow_box_origin[0], x > (flow_box_origin[0] + flow_box_dim[0])),
        quasirandom=cfg.custom.quasirandom,
    )
    thermal_domain.add_constraint(lr_flow_interior, "lr_flow_interior")

    # flow interiror high res near three fin
    logging.info("hr_flow_interior 建立")
    hr_flow_interior = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=geo.geo,
        outvar={"advection_diffusion_theta_f": 0},
        batch_size=cfg.batch_size.InteriorHR,
        criteria=And(x > flow_box_origin[0], x < (flow_box_origin[0] + flow_box_dim[0])),
        quasirandom=cfg.custom.quasirandom,
    )
    thermal_domain.add_constraint(hr_flow_interior, "hr_flow_interior")

    # solid interior
    logging.info("solid_interior 建立")
    solid_interior = PointwiseInteriorConstraint(
        nodes=thermal_nodes,
        geometry=geo.three_fin,
        outvar={"diffusion_theta_s": 0},
        batch_size=cfg.batch_size.SolidInterior,
        lambda_weighting={"diffusion_theta_s": 100.0},
        quasirandom=cfg.custom.quasirandom,
    )
    thermal_domain.add_constraint(solid_interior, "solid_interior")

    # add peak temp monitors for design optimization
    # run only for parameterized cases and in eval mode
    if cfg.custom.parameterized and cfg.run_mode == "eval":
        # define candidate designs
        num_samples = cfg.custom.num_samples
        inference_param_tuple = itertools.product(
            np.linspace(*height_s_range, num_samples),
            np.linspace(*length_s_range, num_samples),
            np.linspace(*thickness_s_range, num_samples),
        )
        for (
            HS_height_s_,
            HS_length_s_,
            HS_thickness_s_,
        ) in inference_param_tuple:
            HS_height_s = float(HS_height_s_)
            HS_length_s = float(HS_length_s_)
            HS_thickness_s = float(HS_thickness_s_)
            specific_param_ranges = {
                fin_height_s: HS_height_s,
                fin_length_s: HS_length_s,
                fin_thickness_s: HS_thickness_s,
            }

            # add metrics for peak temperature
            plane_param_ranges = {**specific_param_ranges}
            metric = (
                "peak_temp"
                + str(HS_height_s)
                + "_"
                + str(HS_length_s)
                + "_"
                + str(HS_thickness_s)
            )
            invar_temp = geo.three_fin.sample_boundary(
                5000,
                criteria=Eq(y, source_origin[1]),
                parameterization=plane_param_ranges,
            )
            peak_temp_monitor = PointwiseMonitor(
                invar_temp,
                output_names=["theta_s"],
                metrics={metric: lambda var: torch.max(var["theta_s"])},
                nodes=thermal_nodes,
            )
            thermal_domain.add_monitor(peak_temp_monitor)
            
    logging.info("開始建立 Solver")
    flow_slv = Solver(cfg, thermal_domain)
    logging.info("Solver 建立完成")

    logging.info("開始 solve")
    flow_slv.solve()
    logging.info("solve 結束")

if __name__ == "__main__":
    run()

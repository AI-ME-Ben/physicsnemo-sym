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

import sys
import torch
from torch.utils.data import DataLoader, Dataset
from sympy import Symbol, Eq, Abs, tanh, Or, And
import numpy as np
import itertools

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_3d import Box, Channel, Plane
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes, Curl
from physicsnemo.sym.eq.pdes.turbulence_zero_eq import ZeroEquation
from physicsnemo.sym.eq.pdes.basic import NormalDotVec, GradNormal
from physicsnemo.sym.eq.pdes.diffusion import Diffusion, DiffusionInterface
from physicsnemo.sym.eq.pdes.advection_diffusion import AdvectionDiffusion
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch

from three_fin_geometry import *

import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(message)s',
    datefmt='%H:%M:%S'
)

@physicsnemo.sym.main(config_path="conf", config_name="conf_flow")
def run(cfg: PhysicsNeMoConfig) -> None:
    # params for simulation
    # fluid params
    nu=0.02
    rho=1.0
    volumetric_flow = 1.0
    inlet_vel = 1.0
    # make navier stokes equations
    #turbulent
    ns = NavierStokes(nu=nu, rho=rho, dim=3, time=False)
    normal_dot_vel = NormalDotVec()
    equation_nodes = ns.make_nodes() + normal_dot_vel.make_nodes()

    # make network arch
    # parameterized
    input_keys = [
        Key("x"),
        Key("y"),
        Key("z"),
        Key("fin_height_s"),
        Key("fin_length_s"),
        Key("fin_thickness_s"),
    ]
    # exact_continuity
    #c = Curl(("a", "b", "c"), ("u", "v", "w"))
    #equation_nodes += c.make_nodes()
    output_keys = [Key("u"), Key("v"), Key("w"), Key("p")]
    flow_net = ModifiedFourierNetArch(
        input_keys=input_keys,
        output_keys=output_keys,
        adaptive_activations=cfg.custom.adaptive_activations,
    )
    
    logging.info("開始建立網路節點")
    flow_nodes = (
        equation_nodes
        + [flow_net.make_node(name="flow_network")]
    )

    geo = ThreeFin(parameterized=cfg.custom.parameterized)
    logging.info("Geo 建立完成")

    # make flow domain
    logging.info("建立 flow domain")
    flow_domain = Domain()

    # inlet
    logging.info("inlet建立")
    constraint_inlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.inlet,
        outvar={"u": inlet_vel, "v": 0, "w": 0},
        batch_size=cfg.batch_size.Inlet,
        criteria=Eq(x, channel_origin[0]),
        lambda_weighting={
            "u": 1.0,
            "v": 1.0,
            "w": 1.0,
        },  # weight zero on edges
        parameterization=geo.pr,
        batch_per_epoch=5000,
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(constraint_inlet, "inlet")

    # outlet
    logging.info("outlet建立")
    constraint_outlet = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.outlet,
        outvar={"p": 0},
        batch_size=cfg.batch_size.Outlet,
        criteria=Eq(x, channel_origin[0] + channel_dim[0]),
        lambda_weighting={"p": 1.0},
        parameterization=geo.pr,
        batch_per_epoch=5000,
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(constraint_outlet, "outlet")

    # no slip for channel walls
    logging.info("no_slip建立")
    no_slip = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.NoSlip,
        lambda_weighting={
            "u": 1.0,
            "v": 1.0,
            "w": 1.0,
        },  # weight zero on edges
        parameterization=geo.pr,
        batch_per_epoch=5000,
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(no_slip, "no_slip")

    # flow interior low res away from three fin
    logging.info("lr_interior建立")
    lr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.InteriorLR,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
        compute_sdf_derivatives=True,
        parameterization=geo.pr,
        batch_per_epoch=5000,
        criteria=Or(x < flow_box_origin[0], x > (flow_box_origin[0] + flow_box_dim[0])),
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(lr_interior, "lr_interior")

    # flow interiror high res near three fin
    logging.info("hr_interior建立")
    hr_interior = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_z": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.InteriorHR,
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
        compute_sdf_derivatives=True,
        parameterization=geo.pr,
        batch_per_epoch=5000,
        criteria=And(x > flow_box_origin[0], x < (flow_box_origin[0] + flow_box_dim[0])),
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(hr_interior, "hr_interior")

    # integral continuity
    logging.info("integral_criteria建立")
    def integral_criteria(invar, params):
        sdf = geo.geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.integral_plane,
        outvar={"normal_dot_vel": volumetric_flow},
        batch_size=cfg.batch_size.num_integral_continuity,
        integral_batch_size=cfg.batch_size.IntegralContinuity,
        criteria=integral_criteria,
        lambda_weighting={"normal_dot_vel": 1.0},
        parameterization={**geo.pr, **{x_pos: (flow_box_origin[0], flow_box_origin[0] + flow_box_dim[0])}},
        fixed_dataset=False,
        num_workers=4,
        quasirandom=cfg.custom.quasirandom,
    )
    flow_domain.add_constraint(integral_continuity, "integral_continuity")
    logging.info("Domain 和約束建立完成")
    
    # add pressure monitor
    invar_inlet_pressure = geo.integral_plane.sample_boundary(
        1024, parameterization={**fixed_param_ranges, **{x_pos: -2}}
    )
    pressure_monitor = PointwiseMonitor(
        invar_inlet_pressure,
        output_names=["p"],
        metrics={"inlet_pressure": lambda var: torch.mean(var["p"])},
        nodes=flow_nodes,
    )
    flow_domain.add_monitor(pressure_monitor)

    # add pressure drop for design optimization
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

            # add metrics for front pressure
            plane_param_ranges = {
                **specific_param_ranges,
                **{x_pos: heat_sink_base_origin[0] - heat_sink_base_dim[0]},
            }
            metric = (
                "front_pressure"
                + str(HS_height_s)
                + "_"
                + str(HS_length_s)
                + "_"
                + str(HS_thickness_s)
            )
            invar_pressure = geo.integral_plane.sample_boundary(
                1024,
                parameterization=plane_param_ranges,
            )
            front_pressure_monitor = PointwiseMonitor(
                invar_pressure,
                output_names=["p"],
                metrics={metric: lambda var: torch.mean(var["p"])},
                nodes=flow_nodes,
            )
            flow_domain.add_monitor(front_pressure_monitor)

            # add metrics for back pressure
            plane_param_ranges = {
                **specific_param_ranges,
                **{x_pos: heat_sink_base_origin[0] + 2 * heat_sink_base_dim[0]},
            }
            metric = (
                "back_pressure"
                + str(HS_height_s)
                + "_"
                + str(HS_length_s)
                + "_"
                + str(HS_thickness_s)
            )
            invar_pressure = geo.integral_plane.sample_boundary(
                1024,
                parameterization=plane_param_ranges,
            )
            back_pressure_monitor = PointwiseMonitor(
                invar_pressure,
                output_names=["p"],
                metrics={metric: lambda var: torch.mean(var["p"])},
                nodes=flow_nodes,
            )
            flow_domain.add_monitor(back_pressure_monitor)

    # 建立 solver
    logging.info("開始建立 Solver")
    flow_slv = Solver(cfg, flow_domain)
    logging.info("Solver 建立完成")
    
    # 執行 solver
    logging.info("開始 solve")
    flow_slv.solve()
    logging.info("solve 結束")

# 主程序
if __name__ == "__main__":
    run()

import numpy as np
import porepy as pp

from multilayer_interface_law import RobinCouplingMultilayer
from multilayer_rt0 import RT0Multilayer

class Flow(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb, model="flow"):

        self.model = model
        self.gb = gb
        self.data = None
        self.data_time = None
        self.assembler = None
        self.assembler_variable = None

        # discretization operator name
        self.discr_name = self.model + "_flux"
        self.discr = pp.RT0
        self.discr_multilayer = RT0Multilayer

        # coupling operator
        self.coupling_name = self.discr_name + "_coupling"
        self.coupling = pp.RobinCoupling
        self.coupling_multilayer = RobinCouplingMultilayer

        # source
        self.source_name = self.model + "_source"
        self.source = pp.DualScalarSource

        # master variable name
        self.variable = self.model + "_variable"
        self.mortar = self.model + "_lambda"

        # post process variables
        self.pressure = "pressure"
        self.flux = "darcy_flux"  # it has to be this one
        self.P0_flux = "P0_darcy_flux"

    # ------------------------------------------------------------------------------#

    def set_data(self, data, data_time):
        self.data = data
        self.data_time = data_time
        alpha = self.data.get("alpha", 2)

        for g, d in self.gb:
            param = {}

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            d["deviation_from_plane_tol"] = 1e-4
            d["is_tangential"] = True
            d["tol"] = data["tol"]

            # assign permeability
            if "fracture" in g.name:
                fracture_aperture = d[pp.STATE]["fracture_aperture"]
                fracture_aperture_initial = d[pp.STATE]["fracture_aperture_initial"]
                ratio = np.power(fracture_aperture/fracture_aperture_initial, alpha+1)

            elif "layer" in g.name:
                layer_aperture = d[pp.STATE]["layer_aperture"]
                layer_aperture_initial = d[pp.STATE]["layer_aperture_initial"]
                ratio = np.power(layer_aperture/layer_aperture_initial, alpha+1)

            else:
                porosity = d[pp.STATE]["porosity"]
                porosity_initial = d[pp.STATE]["porosity_initial"]
                ratio = np.power(porosity/porosity_initial, alpha)

            # no source term is assumed by the user
            param["second_order_tensor"] = self.set_perm(ratio * self.data["k_t"](g), g)
            param["source"] = zeros

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size:
                labels, param["bc_values"] = data["bc"](g, data, data["tol"])
                param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
            else:
                param["bc_values"] = np.zeros(g.num_faces)
                param["bc"] = pp.BoundaryCondition(g, empty, empty)

            pp.initialize_data(g, d, self.model, param)

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            g_slave, g_master = self.gb.nodes_of_edge(e)
            check_P = mg.slave_to_mortar_avg()

            if "fracture" in g_slave.name or "fracture" in g_master.name:
                aperture = self.gb.node_props(g_slave, pp.STATE)["fracture_aperture"]
                aperture_initial = self.gb.node_props(g_slave, pp.STATE)["fracture_aperture_initial"]
                k_n = self.data["k_n"](g_slave if "fracture" in g_slave.name else g_master)
            else:
                aperture = self.gb.node_props(g_slave, pp.STATE)["layer_aperture"]
                aperture_initial = self.gb.node_props(g_slave, pp.STATE)["layer_aperture_initial"]
                k_n = self.data["k_n"](g_slave)

            k = 2 * check_P * (np.power(aperture/aperture_initial, alpha-1) * k_n)
            pp.initialize_data(mg, d, self.model, {"normal_diffusivity": k})

    # ------------------------------------------------------------------------------#

    def update_data(self):
        # NOTE the self.data stores the original values
        alpha = self.data.get("alpha", 2)

        for g, d in self.gb:
            param = {}
            unity = np.ones(g.num_cells)

            # assign permeability
            if "fracture" in g.name:
                fracture_aperture = d[pp.STATE]["fracture_aperture"]
                fracture_aperture_star = d[pp.STATE]["fracture_aperture_star"]
                fracture_aperture_initial = d[pp.STATE]["fracture_aperture_initial"]

                source = (fracture_aperture_star - fracture_aperture) / self.data_time["step"]
                ratio = np.power(fracture_aperture_star/fracture_aperture_initial, alpha+1)

                # fracture aperture and permeability check
                if np.any(fracture_aperture < 0) or np.any(fracture_aperture_star < 0) or np.any(ratio < 0):
                    raise ValueError(str(np.any(fracture_aperture < 0)) + " " +
                                     str(np.any(fracture_aperture_star < 0)) + " " +
                                     str(np.any(ratio < 0)))

            elif "layer" in g.name:
                layer_aperture = d[pp.STATE]["layer_aperture"]
                layer_aperture_star = d[pp.STATE]["layer_aperture_star"]
                layer_aperture_initial = d[pp.STATE]["layer_aperture_initial"]

                source = (layer_aperture_star - layer_aperture) / self.data_time["step"]
                ratio = np.power(layer_aperture_star/layer_aperture_initial, alpha+1)

                # layer aperture and permeability check
                if np.any(layer_aperture < 0) or np.any(layer_aperture_star < 0) or np.any(ratio < 0):
                    raise ValueError(str(np.any(layer_aperture < 0)) + " " +
                                     str(np.any(layer_aperture_star < 0)) + " " +
                                     str(np.any(ratio < 0)))

            else:
                porosity = d[pp.STATE]["porosity"]
                porosity_star = d[pp.STATE]["porosity_star"]
                porosity_initial = d[pp.STATE]["porosity_initial"]

                source = (porosity_star - porosity) / self.data_time["step"]
                ratio = np.power(porosity_star/porosity_initial, alpha)

                # porosity and permeability check
                if np.any(porosity < 0) or np.any(porosity_star < 0) or np.any(ratio < 0):
                    raise ValueError(str(np.any(porosity < 0)) + " " +
                                     str(np.any(porosity_star < 0)) + " " +
                                     str(np.any(ratio < 0)))

            param["second_order_tensor"] = self.set_perm(ratio * self.data["k_t"](g), g)
            param["source"] = g.cell_volumes * source
            d[pp.PARAMETERS][self.model].update(param)

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            g_slave, g_master = self.gb.nodes_of_edge(e)
            check_P = mg.slave_to_mortar_avg()

            if "fracture" in g_slave.name or "fracture" in g_master.name:
                aperture_star = self.gb.node_props(g_slave, pp.STATE)["fracture_aperture_star"]
                aperture_initial = self.gb.node_props(g_slave, pp.STATE)["fracture_aperture_initial"]
                k_n = self.data["k_n"](g_slave if "fracture" in g_slave.name else g_master)
            else:
                aperture_star = self.gb.node_props(g_slave, pp.STATE)["layer_aperture_star"]
                aperture_initial = self.gb.node_props(g_slave, pp.STATE)["layer_aperture_initial"]
                k_n = self.data["k_n"](g_slave)

            k = 2 * check_P * (np.power(aperture_star/aperture_initial, alpha-1) * k_n)
            d[pp.PARAMETERS][self.model].update({"normal_diffusivity": k})

    # ------------------------------------------------------------------------------#

    def shape(self):
        return self.gb.num_cells() + self.gb.num_faces() + self.gb.num_mortar_cells()

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # set the discretization for the grids
        for g, d in self.gb:
            if g.dim == self.gb.dim_max():
                discr = self.discr(self.model)
            else:
                discr = self.discr_multilayer(self.model)

            source = self.source(self.model)

            d[pp.PRIMARY_VARIABLES] = {self.variable: {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {self.variable: {self.discr_name: discr,
                                                    self.source_name: source}}
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # define the interface terms to couple the grids
        for e, d in self.gb.edges():
            g_slave, g_master = self.gb.nodes_of_edge(e)

            # retrive the discretization of the master and slave grids
            discr_master = self.gb.node_props(g_master, pp.DISCRETIZATION)[self.variable][self.discr_name]
            discr_slave = self.gb.node_props(g_slave, pp.DISCRETIZATION)[self.variable][self.discr_name]

            if g_master.dim == self.gb.dim_max():
                # classical 2d-1d/3d-2d coupling condition
                coupling = self.coupling(self.model, discr_master, discr_slave)
            else:
                # the multilayer coupling condition
                coupling = self.coupling_multilayer(self.model, discr_master, discr_slave)

            d[pp.PRIMARY_VARIABLES] = {self.mortar: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                self.coupling_name: {
                    g_slave: (self.variable, self.discr_name),
                    g_master: (self.variable, self.discr_name),
                    e: (self.mortar, coupling),
                }
            }
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # assembler
        self.assembler = pp.Assembler(self.gb)
        self.assembler.discretize()
        return self.assembler.assemble_matrix_rhs()

    # ------------------------------------------------------------------------------#

    def extract(self, x):
        self.assembler.distribute_variable(x)

        discr = self.discr(self.model)
        for g, d in self.gb:
            var = d[pp.STATE][self.variable]
            d[pp.STATE][self.pressure] = discr.extract_pressure(g, var, d)
            d[pp.STATE][self.flux] = discr.extract_flux(g, var, d)

        # export the P0 flux reconstruction
        pp.project_flux(self.gb, discr, self.flux, self.P0_flux, self.mortar)

    # ------------------------------------------------------------------------------#

    def set_perm(self, k, g):
        if g.dim == 1:
            return pp.SecondOrderTensor(kxx=k, kyy=1, kzz=1)
        elif g.dim == 2:
            return pp.SecondOrderTensor(kxx=k, kyy=k, kzz=1)
        elif g.dim == 3:
            return pp.SecondOrderTensor(kxx=k, kyy=k, kzz=k)

    # ------------------------------------------------------------------------------#

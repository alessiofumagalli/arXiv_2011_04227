import numpy as np
import porepy as pp

from multilayer_interface_law import RobinCouplingMultilayer
from multilayer_tpfa import ImplicitTpfaMultilayer
from multilayer_hyperbolic_interface_laws import ImplicitUpwindCouplingMultilayer

from porepy.utils.derived_discretizations import implicit_euler as IE_discretizations

class Heat(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb, model="heat"):

        self.model = model
        self.gb = gb
        self.data = None
        self.data_time = None
        self.assembler = None

        # discretization operator name
        self.diff_name = self.model + "_diff"
        self.diff = IE_discretizations.ImplicitTpfa
        self.diff_multilayer = ImplicitTpfaMultilayer

        self.adv_name = self.model + "_adv"
        self.adv = IE_discretizations.ImplicitUpwind

        self.mass_name_lhs = self.model + "_mass_lhs"
        self.mass_name_rhs = self.model + "_mass_rhs"
        self.mass = IE_discretizations.ImplicitMassMatrix

        # coupling operator
        self.coupling_diff_name = self.diff_name + "_coupling"
        self.coupling_diff = pp.RobinCoupling
        self.coupling_diff_multilayer = RobinCouplingMultilayer

        self.coupling_adv_name = self.adv_name + "_coupling"
        self.coupling_adv = IE_discretizations.ImplicitUpwindCoupling
        self.coupling_adv_multilayer = ImplicitUpwindCouplingMultilayer

        # master variable name
        self.variable = self.model + "_variable"
        self.mortar_diff = self.diff_name + "_lambda"
        self.mortar_adv = self.adv_name + "_lambda"

        # post process variables
        self.scalar = self.model + "_scalar"
        self.flux = "darcy_flux"

    # ------------------------------------------------------------------------------#

    def set_data(self, data, data_time):
        self.data = data
        self.data_time = data_time

        for g, d in self.gb:
            param_diff = {}
            param_adv = {}
            param_mass_lhs = {}
            param_mass_rhs = {}

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            d["Aavatsmark_transmissibilities"] = True
            d["tol"] = data["tol"]

            # assign permeability
            if "fracture" in g.name:
                diff = data["l_w"] * d[pp.STATE]["fracture_aperture"]
            elif "layer" in g.name:
                diff = data["l_w"] * d[pp.STATE]["layer_aperture"]
            else:
                porosity = d[pp.STATE]["porosity"]
                diff = np.power(self.data["l_w"], porosity)*np.power(self.data["l_s"], 1.-porosity)

            param_diff["second_order_tensor"] = pp.SecondOrderTensor(diff)

            # compute the effective thermal capacity, keeping in mind that the fracture
            # contains only water

            vol_star = d[pp.STATE]["porosity_star"] + d[pp.STATE]["fracture_aperture_star"] +\
                       d[pp.STATE]["layer_aperture_star"]

            vol_old = d[pp.STATE]["porosity_old"] + d[pp.STATE]["fracture_aperture_old"] + \
                      d[pp.STATE]["layer_aperture_old"]

            c_star = self.data["rc_w"] * vol_star + self.data["rc_s"] * (1 - d[pp.STATE]["porosity_star"])
            c_old = self.data["rc_w"] * vol_old + self.data["rc_s"] * (1 - d[pp.STATE]["porosity_old"])

            param_mass_lhs["mass_weight"] = c_star
            param_mass_rhs["mass_weight"] = c_old

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size:
                labels_diff, labels_adv, bc_val = data["bc"](g, data, data["tol"])
                param_diff["bc"] = pp.BoundaryCondition(g, b_faces, labels_diff)
                param_adv["bc"] = pp.BoundaryCondition(g, b_faces, labels_adv)
            else:
                bc_val = np.zeros(g.num_faces)
                param_diff["bc"] = pp.BoundaryCondition(g, empty, empty)
                param_adv["bc"] = pp.BoundaryCondition(g, empty, empty)

            param_diff["bc_values"] = bc_val
            param_adv["bc_values"] = bc_val

            param_diff["time_step"] = data_time["step"]
            param_adv["time_step"] = data_time["step"]

            param_adv["advection_weight"] = data["rc_w"]

            models = [self.diff_name, self.adv_name, self.mass_name_lhs, self.mass_name_rhs]
            params = [param_diff, param_adv, param_mass_lhs, param_mass_rhs]
            for model, param in zip(models, params):
                pp.initialize_data(g, d, model, param)

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            g_slave, g_master = self.gb.nodes_of_edge(e)
            check_P = mg.slave_to_mortar_avg()

            if "fracture" in g_slave.name or "fracture" in g_master.name:
                aperture = self.gb.node_props(g_slave, pp.STATE)["fracture_aperture"]
            else:
                aperture = self.gb.node_props(g_slave, pp.STATE)["layer_aperture"]

            l = 2 * check_P * (self.data["l_w"] / aperture)

            models = [self.diff_name, self.adv_name]
            params = [{"normal_diffusivity": l}, {"advection_weight": data["rc_w"]}]
            for model, param in zip(models, params):
                pp.initialize_data(mg, d, model, param)

    # ------------------------------------------------------------------------------#

    def update_data(self):
        # NOTE the self.data stores the original values

        for g, d in self.gb:
            param_diff = {}

            # assign permeability
            if "fracture" in g.name:
                l = self.data["l_w"] * d[pp.STATE]["fracture_aperture_star"]
            elif "layer" in g.name:
                l = self.data["l_w"] * d[pp.STATE]["layer_aperture_star"]
            else:
                poro_star = d[pp.STATE]["porosity_star"]
                l = np.power(self.data["l_w"], poro_star)*np.power(self.data["l_s"], 1-poro_star)

            param_diff["second_order_tensor"] = pp.SecondOrderTensor(l)
            d[pp.PARAMETERS][self.diff_name].update(param_diff)

            # compute the effective thermal capacity, keeping in mind that the fracture
            # contains only water

            vol_star = d[pp.STATE]["porosity_star"] + d[pp.STATE]["fracture_aperture_star"] +\
                       d[pp.STATE]["layer_aperture_star"]

            vol_old = d[pp.STATE]["porosity_old"] + d[pp.STATE]["fracture_aperture_old"] + \
                      d[pp.STATE]["layer_aperture_old"]

            c_star = self.data["rc_w"] * vol_star + self.data["rc_s"] * (1 - d[pp.STATE]["porosity_star"])
            c_old = self.data["rc_w"] * vol_old + self.data["rc_s"] * (1 - d[pp.STATE]["porosity_old"])

            d[pp.PARAMETERS][self.mass_name_lhs].update({"mass_weight": c_star})
            d[pp.PARAMETERS][self.mass_name_rhs].update({"mass_weight": c_old})

        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            g_slave, g_master = self.gb.nodes_of_edge(e)
            check_P = d["mortar_grid"].slave_to_mortar_avg()

            if "fracture" in g_slave.name or "fracture" in g_master.name:
                aperture_star = self.gb.node_props(g_slave, pp.STATE)["fracture_aperture_star"]
            else:
                aperture_star = self.gb.node_props(g_slave, pp.STATE)["layer_aperture_star"]

            l = 2 * check_P * (self.data["l_w"] / aperture_star)
            d[pp.PARAMETERS][self.diff_name].update({"normal_diffusivity": l})

    # ------------------------------------------------------------------------------#

    def shape(self):
        return self.gb.num_cells() + 2*self.gb.num_mortar_cells()

    # ------------------------------------------------------------------------------#

    def set_flux(self, flux_name, mortar_name):
        for _, d in self.gb:
            d[pp.PARAMETERS][self.adv_name][self.flux] = d[pp.STATE][flux_name]

        for _, d in self.gb.edges():
            d[pp.PARAMETERS][self.adv_name][self.flux] = d[pp.STATE][mortar_name]

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # set the discretization for the grids
        for g, d in self.gb:
            if g.dim == self.gb.dim_max():
                diff = self.diff(self.diff_name)
            else:
                diff = self.diff_multilayer(self.diff_name)

            adv = self.adv(self.adv_name)
            mass_lhs = self.mass(self.mass_name_lhs, "temperature_old")
            mass_rhs = self.mass(self.mass_name_rhs, "temperature_old")

            d[pp.PRIMARY_VARIABLES] = {self.variable: {"cells": 1}}
            d[pp.DISCRETIZATION] = {self.variable: {self.diff_name: diff,
                                                    self.adv_name: adv,
                                                    self.mass_name_lhs: mass_lhs,
                                                    self.mass_name_rhs: mass_rhs}}
            d[pp.DISCRETIZATION_MATRICES] = {self.diff_name: {},
                                             self.adv_name: {},
                                             self.mass_name_lhs: {},
                                             self.mass_name_rhs: {}}

        for e, d in self.gb.edges():

            g_slave, g_master = self.gb.nodes_of_edge(e)

            # retrive the discretization of the master and slave grids for the diffusion
            diff_master = self.gb.node_props(g_master, pp.DISCRETIZATION)[self.variable][self.diff_name]
            diff_slave = self.gb.node_props(g_slave, pp.DISCRETIZATION)[self.variable][self.diff_name]

            if g_master.dim == self.gb.dim_max():
                # classical 2d-1d/3d-2d coupling condition
                coupling_diff = self.coupling_diff(self.diff_name, diff_master, diff_slave)
                coupling_adv = self.coupling_adv(self.adv_name)
            else:
                # the multilayer coupling condition
                coupling_diff = self.coupling_diff_multilayer(self.diff_name, diff_master, diff_slave)
                coupling_adv = self.coupling_adv_multilayer(self.adv_name)

            d[pp.PRIMARY_VARIABLES] = {self.mortar_diff: {"cells": 1},
                                       self.mortar_adv: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                self.coupling_diff_name: {
                    g_slave: (self.variable, self.diff_name),
                    g_master: (self.variable, self.diff_name),
                    e: (self.mortar_diff, coupling_diff),
                },
                self.coupling_adv_name: {
                    g_slave: (self.variable, self.adv_name),
                    g_master: (self.variable, self.adv_name),
                    e: (self.mortar_adv, coupling_adv),
                }
            }
            d[pp.DISCRETIZATION_MATRICES] = {self.diff_name: {},
                                             self.adv_name: {},
                                             self.mass_name_lhs: {},
                                             self.mass_name_rhs: {}}

        # assembler
        self.assembler = pp.Assembler(self.gb)
        self.assembler.discretize()
        block_A, block_b = self.assembler.assemble_matrix_rhs(add_matrices=False)

        # unpack the matrices just computed
        coupling_diff_name = self.coupling_diff_name + (
            "_" + self.mortar_diff + "_" + self.variable + "_" + self.variable
        )
        coupling_adv_name = self.coupling_adv_name + (
            "_" + self.mortar_adv + "_" + self.variable + "_" + self.variable
        )
        diff_name = self.diff_name + "_" + self.variable
        adv_name = self.adv_name + "_" + self.variable
        mass_name_lhs = self.mass_name_lhs + "_" + self.variable
        mass_name_rhs = self.mass_name_rhs + "_" + self.variable

        # extract the matrices
        M = block_A[mass_name_lhs]
        m = block_b[mass_name_rhs]

        if self.gb.size() > 1:
            A = block_A[diff_name] + block_A[coupling_diff_name] +\
                block_A[adv_name] + block_A[coupling_adv_name]
            b = block_b[diff_name] + block_b[coupling_diff_name] +\
                block_b[adv_name] + block_b[coupling_adv_name]
        else:
            A = block_A[diff_name] + block_A[adv_name]
            b = block_b[diff_name] + block_b[adv_name]

        return A + M, b + m

    # ------------------------------------------------------------------------------#

    def extract(self, x, name=None):
        self.assembler.distribute_variable(x)
        if name is None:
            name = self.scalar
        for _, d in self.gb:
            d[pp.STATE][name] = d[pp.STATE][self.variable]

    # ------------------------------------------------------------------------------#

import scipy.sparse as sps
import numpy as np
import porepy as pp

from multilayer_interface_law import RobinCouplingMultiLayer
from multilayer_rt0 import RT0Multilayer

# ------------------------------------------------------------------------------#

def data_flow(gb, model, data, bc_flag):
    tol = data["tol"]

    model_data = model + "_data"

    for g, d in gb:
        param = {}

        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        d["is_tangential"] = True
        d["tol"] = tol

        # assign permeability
        if "fault" in g.name:
            data_fault = data["fault"]
            kxx = data_fault["kf_t"] * unity
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=1, kzz=1)
            aperture = data_fault["aperture"] * unity

        elif "layer" in g.name:
            data_layer = data["layer"]
            kxx = data_layer["kf_t"] * unity
            perm = pp.SecondOrderTensor(kxx=kxx, kyy=1, kzz=1)
            aperture = data_layer["aperture"] * unity

        else:
            kxx = data["k"] * unity
            if g.dim == 2:
                perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=1)
            else:
                perm = pp.SecondOrderTensor(kxx=kxx, kyy=kxx, kzz=kxx)
            aperture = unity

        param["second_order_tensor"] = perm
        param["aperture"] = aperture

        # source
        param["source"] = zeros

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size:
            labels, bc_val = bc_flag(g, data, tol)
            param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
        else:
            bc_val = np.zeros(g.num_faces)
            param["bc"] = pp.BoundaryCondition(g, empty, empty)

        param["bc_values"] = bc_val

        pp.initialize_data(g, d, model_data, param)

    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]

        if "layer" in g_l.name:
            data_interface = data["layer"]
        else:
            data_interface = data["fault"]

        mg = d["mortar_grid"]
        check_P = mg.slave_to_mortar_avg()

        aperture = gb.node_props(g_l, pp.PARAMETERS)[model_data]["aperture"]
        kn = 2 * data_interface["kf_n"] / (check_P * aperture)

        pp.initialize_data(mg, d, model_data, {"normal_diffusivity": kn})
        d[pp.DISCRETIZATION_MATRICES] = {model_data: {}}

    return model_data


# ------------------------------------------------------------------------------#


def flow(gb, param, bc_flag):

    model = "flow"

    model_data = data_flow(gb, model, param, bc_flag)

    # discretization operator name
    flux_id = "flux"

    # master variable name
    variable = "flow_variable"
    mortar = "lambda_" + variable

    # post process variables
    pressure = "pressure"
    flux = "darcy_flux"  # it has to be this one

    # save variable name for the advection-diffusion problem
    param["pressure"] = pressure
    param["flux"] = flux
    param["mortar_flux"] = mortar

    # define the dof and discretization for the grids
    for g, d in gb:
        d[pp.PRIMARY_VARIABLES] = {variable: {"cells": 1, "faces": 1}}
        if g.dim == gb.dim_max():
            discr = pp.RT0(model_data)
        else:
            discr = RT0Multilayer(model_data)
        d[pp.DISCRETIZATION] = {variable: {flux_id: discr}}

    # define the interface terms to couple the grids
    for e, d in gb.edges():
        g_slave, g_master = gb.nodes_of_edge(e)
        d[pp.PRIMARY_VARIABLES] = {mortar: {"cells": 1}}

        # retrive the discretization of the master and slave grids
        discr_master = gb.node_props(g_master, pp.DISCRETIZATION)[variable][flux_id]
        discr_slave = gb.node_props(g_slave, pp.DISCRETIZATION)[variable][flux_id]

        if g_master.dim == gb.dim_max():
            # classical 2d-1d/3d-2d coupling condition
            coupling = pp.RobinCoupling(model_data, discr_master, discr_slave)

            d[pp.COUPLING_DISCRETIZATION] = {
                flux: {
                    g_slave: (variable, flux_id),
                    g_master: (variable, flux_id),
                    e: (mortar, coupling),
                }
            }
        elif g_master.dim < gb.dim_max():
            # the multilayer coupling condition
            coupling = RobinCouplingMultiLayer(model_data, discr_master, discr_slave)

            d[pp.COUPLING_DISCRETIZATION] = {
                flux: {
                    g_slave: (variable, flux_id),
                    g_master: (variable, flux_id),
                    e: (mortar, coupling),
                }
            }

    # empty the matrices
    for g, d in gb:
        d[pp.DISCRETIZATION_MATRICES][model_data] = {}

    for e, d in gb.edges():
        d[pp.DISCRETIZATION_MATRICES][model_data] = {}

    # assembler
    assembler = pp.Assembler(gb)
    assembler.discretize()

    A, b = assembler.assemble_matrix_rhs()
    x = sps.linalg.spsolve(A, b)

    assembler.distribute_variable(x)
    for g, d in gb:
        var = d[pp.STATE][variable]
        d[pp.STATE][pressure] = discr.extract_pressure(g, var, d)
        d[pp.STATE][flux] = discr.extract_flux(g, var, d)

    # export the P0 flux reconstruction
    P0_flux = "P0_flux"
    param["P0_flux"] = P0_flux
    pp.project_flux(gb, discr, flux, P0_flux, mortar)

    # identification of layer and fault
    for g, d in gb:
        # save the identification of the fault
        if "fault" in g.name:
            d[pp.STATE]["fault"] = np.ones(g.num_cells)
            d[pp.STATE]["layer"] = np.zeros(g.num_cells)
        # save the identification of the layer
        elif "layer" in g.name:
            d[pp.STATE]["fault"] = np.zeros(g.num_cells)
            half_cells = int(g.num_cells / 2)
            d[pp.STATE]["layer"] = np.hstack((np.ones(half_cells), 2 * np.ones(half_cells)))
        # save zero for the other cases
        else:
            d[pp.STATE]["fault"] = np.zeros(g.num_cells)
            d[pp.STATE]["layer"] = np.zeros(g.num_cells)

    save = pp.Exporter(gb, "solution", folder_name=param["folder"])
    save.write_vtk([pressure, P0_flux, "fault", "layer"])

# ------------------------------------------------------------------------------#

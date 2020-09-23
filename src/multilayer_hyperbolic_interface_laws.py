"""
Module of coupling laws for hyperbolic equations.
"""

import numpy as np
import scipy.sparse as sps

import porepy as pp
import porepy.numerics.interface_laws.abstract_interface_law


class ImplicitUpwindCouplingMultilayer(pp.UpwindCoupling):
    def __init__(self, keyword):
        super(ImplicitUpwindCouplingMultilayer, self).__init__(keyword)

    def assemble_matrix_rhs(
        self, g_master, g_slave, data_master, data_slave, data_edge, matrix
    ):
        """
        Construct the matrix (and right-hand side) for the coupling conditions.
        Note: the right-hand side is not implemented now.

        Parameters:
            g_master: grid of higher dimension
            g_slave: grid of lower dimension
            data_master: dictionary which stores the data for the higher dimensional
                grid
            data_slave: dictionary which stores the data for the lower dimensional
                grid
            data_edge: dictionary which stores the data for the edges of the grid
                bucket
            matrix: Uncoupled discretization matrix.

        Returns:
            cc: block matrix which store the contribution of the coupling
                condition. See the abstract coupling class for a more detailed
                description.

        """

        # Normal component of the velocity from the higher dimensional grid

        # @ALL: This should perhaps be defined by a globalized keyword
        parameter_dictionary_master = data_master[pp.PARAMETERS]
        parameter_dictionary_slave = data_slave[pp.PARAMETERS]
        lam_flux = data_edge[pp.PARAMETERS][self.keyword]["darcy_flux"]
        dt = parameter_dictionary_master[self.keyword]["time_step"]
        w_master = (
            parameter_dictionary_master.expand_scalars(
                g_master.num_cells, self.keyword, ["advection_weight"]
            )[0]
            * dt
        )
        w_slave = (
            parameter_dictionary_slave.expand_scalars(
                g_slave.num_cells, self.keyword, ["advection_weight"]
            )[0]
            * dt
        )
        # Retrieve the number of degrees of both grids
        # Create the block matrix for the contributions
        g_m = data_edge["mortar_grid"]

        # We know the number of dofs from the master and slave side from their
        # discretizations
        dof = np.array([matrix[0, 0].shape[1], matrix[1, 1].shape[1], g_m.num_cells])
        cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
        cc = cc.reshape((3, 3))

        # Projection from mortar to upper dimenional faces
        hat_P_avg = g_m.master_to_mortar_avg()
        # Projection from mortar to lower dimensional cells
        check_P_avg = g_m.slave_to_mortar_avg()

        # The mortars always points from upper to lower, so we don't flip any
        # signs

        # Find upwind weighting. if flag is True we use the upper weights
        # if flag is False we use the lower weighs
        flag = (lam_flux > 0).astype(np.float)
        not_flag = 1 - flag

        # assemble matrices
        # Transport out of upper equals lambda
        cc[0, 2] = hat_P_avg.T

        # transport out of lower is -lambda
        cc[1, 2] = -check_P_avg.T  # * sps.diags((1 - flag))

        # Discretisation of mortars
        # If fluid flux(lam_flux) is positive we use the upper value as weight,
        # i.e., T_masterat * fluid_flux = lambda.
        # We set cc[2, 0] = T_masterat * fluid_flux
        cc[2, 0] = sps.diags(lam_flux * flag) * hat_P_avg * sps.diags(w_master)

        # If fluid flux is negative we use the lower value as weight,
        # i.e., T_check * fluid_flux = lambda.
        # we set cc[2, 1] = T_check * fluid_flux
        cc[2, 1] = sps.diags(lam_flux * not_flag) * check_P_avg * sps.diags(w_slave)

        # The rhs of T * fluid_flux = lambda
        # Recover the information for the grid-grid mapping
        cc[2, 2] = -sps.eye(g_m.num_cells)

        if data_master["node_number"] == data_slave["node_number"]:
            # All contributions to be returned to the same block of the
            # global matrix in this case
            cc = np.array([np.sum(cc, axis=(0, 1))])

        # rhs is zero
        rhs = np.squeeze([np.zeros(dof[0]), np.zeros(dof[1]), np.zeros(dof[2])])
        matrix += cc
        return matrix, rhs

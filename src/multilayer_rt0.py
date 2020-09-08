# -*- coding: utf-8 -*-
"""

@author: fumagalli, alessio
"""

import scipy.sparse as sps
import porepy as pp

class RT0Multilayer(pp.RT0):
    """
    Modification of the standard RT0 class to allow for an extra coefficient in the boundary reconstruction operators.
    In this situation it will be essentially needed to couple two equi-dimensional domains.
    This implementation is done to not touch the core, it might be revised in the future.
    """
    def __init__(self, keyword: str) -> None:
        super(RT0Multilayer, self).__init__(keyword)

    def assemble_int_bound_source_1(
        self, g, data, data_edge, grid_swap, cc, matrix, rhs, self_ind, coeff = 1
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a source term.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            coef (float): scalar coefficient to multiply the resulting matrix before
                the summing into cc. In this case is 1, related to the standard Robin interface law.

        """
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.master_to_mortar_avg()
        else:
            proj = mg.slave_to_mortar_avg()

        A = proj.T
        shape = (g.num_faces, A.shape[1])
        cc[self_ind, 2] += coeff * sps.bmat([[sps.csr_matrix(shape)], [A]])

    def assemble_int_bound_pressure_cell_1(
        self, g, data, data_edge, grid_swap, cc, matrix, rhs, self_ind, coeff = 1
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the cell pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            coefficient (float): scalar coefficient to multiply the resulting matrix before
                the summing into cc. In this case is 1, related to the standard Robin interface law.

        """
        mg = data_edge["mortar_grid"]
        # proj = mg.slave_to_mortar_avg()

        if grid_swap:
            proj = mg.master_to_mortar_avg()
        else:
            proj = mg.slave_to_mortar_avg()

        A = proj.T
        shape = (g.num_faces, A.shape[1])

        cc[2, self_ind] -= coeff * sps.bmat([[sps.csr_matrix(shape)], [A]]).T

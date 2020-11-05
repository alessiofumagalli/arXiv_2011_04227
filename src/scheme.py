import numpy as np
import scipy.sparse as sps
import porepy as pp

from flow import Flow
from transport import Transport
from reaction import Reaction
from porosity import Porosity
from fracture_aperture import FractureAperture
from layer_aperture import LayerAperture
from heat import Heat

class Scheme(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb):
        self.gb = gb

        # -- flow -- #
        self.discr_flow = Flow(gb)

        shape = self.discr_flow.shape()
        self.flux_pressure = np.zeros(shape)

        # -- temperature -- #
        self.discr_temperature = Heat(gb)

        # -- solute and precipitate -- #
        self.discr_solute_advection_diffusion = Transport(gb)
        self.discr_solute_precipitate_reaction = Reaction(gb)

        # -- porosity -- #
        self.discr_porosity = Porosity(gb)

        # -- fracture aperture -- #
        self.discr_fracture_aperture = FractureAperture(gb, "fracture_aperture")

        # -- layer porosity and aperture -- #
        self.discr_layer_porosity = Porosity(gb, "layer_porosity")
        self.discr_layer_aperture = LayerAperture(gb, "layer_aperture")

        # the actual time of the simulation
        self.time = 0

    # ------------------------------------------------------------------------------#

    def compute_precipitate_temperature_star(self):
        for g, d in self.gb:
            d[pp.STATE]["precipitate_star"] = 2*d[pp.STATE]["precipitate"] - d[pp.STATE]["precipitate_old"]
            d[pp.STATE]["temperature_star"] = 2*d[pp.STATE]["temperature"] - d[pp.STATE]["temperature_old"]

    # ------------------------------------------------------------------------------#

    def compute_porosity_aperture_star(self):
        dim_max = self.gb.dim_max()
        # only the rock matrix has the variable porosity
        for g in self.gb.grids_of_dimension(dim_max):
            d = self.gb.node_props(g)
            d[pp.STATE]["porosity_star"] = self.discr_porosity.step(d[pp.STATE]["porosity_old"],
                                                                    d[pp.STATE]["precipitate_star"],
                                                                    d[pp.STATE]["precipitate_old"])

        # only the fracture and the layer have the variable aperture
        for g in self.gb.grids_of_dimension(dim_max-1):
            d = self.gb.node_props(g)
            if "fracture" in g.name:
                d[pp.STATE]["fracture_aperture_star"] = self.discr_fracture_aperture.step(d[pp.STATE]["fracture_aperture_old"],
                                                                                          d[pp.STATE]["precipitate_star"],
                                                                                          d[pp.STATE]["precipitate_old"])
            if "layer" in g.name:
                d[pp.STATE]["layer_aperture_star"] = d[pp.STATE]["layer_aperture_old"]
                d[pp.STATE]["layer_porosity_star"] = self.discr_layer_porosity.step(d[pp.STATE]["layer_porosity_old"],
                                                                                    d[pp.STATE]["precipitate_star"],
                                                                                    d[pp.STATE]["precipitate_old"])

    # ------------------------------------------------------------------------------#

    def compute_flow(self):
        A, b = self.discr_flow.matrix_rhs()
        x = sps.linalg.spsolve(A, b)
        if not np.all(np.isfinite(x)):
            raise ValueError
        self.discr_flow.extract(x)

    # ------------------------------------------------------------------------------#

    def compute_temperature(self):
        A, b = self.discr_temperature.matrix_rhs()
        x = sps.linalg.spsolve(A, b)
        if not np.all(np.isfinite(x)):
            raise ValueError
        self.discr_temperature.extract(x, "temperature")

    # ------------------------------------------------------------------------------#

    def compute_solute_precipitate_advection_diffusion(self):
        A, b = self.discr_solute_advection_diffusion.matrix_rhs()
        x = sps.linalg.spsolve(A, b)
        if not np.all(np.isfinite(x)):
            raise ValueError
        self.discr_solute_advection_diffusion.extract(x, "solute_half")

    # ------------------------------------------------------------------------------#

    def correct_precipitate_with_star(self):
        for g, d in self.gb:
            vol_star = d[pp.STATE]["porosity_star"] + d[pp.STATE]["fracture_aperture_star"] +\
                       d[pp.STATE]["layer_porosity_star"]

            vol_old = d[pp.STATE]["porosity_old"] + d[pp.STATE]["fracture_aperture_old"] + \
                      d[pp.STATE]["layer_porosity_old"]

            d[pp.STATE]["precipitate_half"] = d[pp.STATE]["precipitate_old"] * (vol_old / vol_star)

    # ------------------------------------------------------------------------------#

    def compute_solute_precipitate_rection(self):
        for g, d in self.gb:
            d[pp.STATE]["solute_star_star"], d[pp.STATE]["precipitate_star_star"] = \
                self.discr_solute_precipitate_reaction.step(d[pp.STATE]["solute_half"],
                                                            d[pp.STATE]["precipitate_half"],
                                                            d[pp.STATE]["temperature"])

    # ------------------------------------------------------------------------------#

    def compute_porosity_aperture(self):
        dim_max = self.gb.dim_max()
        # only the rock matrix has the variable porosity
        for g in self.gb.grids_of_dimension(dim_max):
            d = self.gb.node_props(g)
            d[pp.STATE]["porosity"] = self.discr_porosity.step(d[pp.STATE]["porosity"],
                                                               d[pp.STATE]["precipitate_star_star"],
                                                               d[pp.STATE]["precipitate_old"])

        # only the fracture and the layer have the variable aperture
        for g in self.gb.grids_of_dimension(dim_max-1):
            d = self.gb.node_props(g)
            if "fracture" in g.name:
                d[pp.STATE]["fracture_aperture"] = self.discr_fracture_aperture.step(d[pp.STATE]["fracture_aperture"],
                                                                                     d[pp.STATE]["precipitate_star_star"],
                                                                                     d[pp.STATE]["precipitate_old"])

            if "layer" in g.name:
                flux_fracture = np.zeros(g.num_cells)
                solute_fracture = np.zeros(g.num_cells)
                porosity = np.zeros(g.num_cells)
                for e, d_e in self.gb.edges_of_node(g):
                    g_m = d_e["mortar_grid"]
                    g_l, g_h = self.gb.nodes_of_edge(e)
                    if g_h.dim < self.gb.dim_max():
                        # save the flux from the fracture
                        flux_fracture = -g_m.master_to_mortar_avg().T * d_e[pp.STATE][self.discr_flow.mortar] / g.cell_volumes
                        d_fracture = self.gb.node_props(g_l if "fracture" in g_l.name else g_h)
                        solute_fracture = g_m.slave_to_mortar_avg() * d_fracture[pp.STATE]["solute"]
                    else:
                        cell_cell = g_m.mortar_to_master_int().T * g_h.cell_faces
                        # the cells are already ordered according to the layer ordering of the cells given by the mortar
                        # map
                        interface_cells = cell_cell.indices
                        d_h = self.gb.node_props(g_h)

                        porosity = d_h[pp.STATE]["porosity"][interface_cells]
                        temperature = d_h[pp.STATE]["temperature"][interface_cells]
                        lmbda = self.discr_solute_precipitate_reaction.data["lambda"](temperature)

                d[pp.STATE]["layer_aperture"] = self.discr_layer_aperture.step(flux_fracture, porosity, lmbda, self.time, solute_fracture)

                d[pp.STATE]["layer_porosity"] = self.discr_layer_porosity.step(d[pp.STATE]["layer_porosity"],
                                                                               d[pp.STATE]["precipitate_star_star"],
                                                                               d[pp.STATE]["precipitate_old"])

    # ------------------------------------------------------------------------------#

    def correct_precipitate_solute(self):
        for g, d in self.gb:
            vol_star = d[pp.STATE]["porosity_star"] + d[pp.STATE]["fracture_aperture_star"] +\
                       d[pp.STATE]["layer_porosity_star"]

            vol = d[pp.STATE]["porosity"] + d[pp.STATE]["fracture_aperture"] + \
                  d[pp.STATE]["layer_porosity"]

            d[pp.STATE]["solute"] = d[pp.STATE]["solute_star_star"] * (vol_star / vol)
            d[pp.STATE]["precipitate"] = d[pp.STATE]["precipitate_star_star"] * (vol_star / vol)

    # ------------------------------------------------------------------------------#

    def set_old_variables(self):
        for g, d in self.gb:
            d[pp.STATE]["temperature_old"] = d[pp.STATE]["temperature"].copy()
            d[pp.STATE]["solute_old"] = d[pp.STATE]["solute"].copy()
            d[pp.STATE]["precipitate_old"] = d[pp.STATE]["precipitate"].copy()
            d[pp.STATE]["porosity_old"] = d[pp.STATE]["porosity"].copy()
            d[pp.STATE]["fracture_aperture_old"] = d[pp.STATE]["fracture_aperture"].copy()
            d[pp.STATE]["layer_aperture_old"] = d[pp.STATE]["layer_aperture"].copy()
            d[pp.STATE]["layer_porosity_old"] = d[pp.STATE]["layer_porosity"].copy()

    # ------------------------------------------------------------------------------#

    def set_data(self, param):

        for g, d in self.gb:
            data = param["temperature"]["initial"]
            d[pp.STATE]["temperature"] = data(g, param, param["tol"])
            d[pp.STATE]["temperature_star"] = d[pp.STATE]["temperature"].copy()

            data = param["solute_advection_diffusion"]["initial_solute"]
            d[pp.STATE]["solute"] = data(g, param, param["tol"])

            data = param["solute_advection_diffusion"]["initial_precipitate"]
            d[pp.STATE]["precipitate"] = data(g, param, param["tol"])
            d[pp.STATE]["precipitate_star"] = d[pp.STATE]["precipitate"].copy()

            data = param["porosity"]["initial"]
            d[pp.STATE]["porosity_initial"] = data(g, param, param["tol"])
            d[pp.STATE]["porosity_star"] = d[pp.STATE]["porosity_initial"].copy()
            d[pp.STATE]["porosity"] = d[pp.STATE]["porosity_initial"].copy()

            data = param["fracture_aperture"]["initial"]
            d[pp.STATE]["fracture_aperture_initial"] = data(g, param, param["tol"])
            d[pp.STATE]["fracture_aperture_star"] = d[pp.STATE]["fracture_aperture_initial"].copy()
            d[pp.STATE]["fracture_aperture"] = d[pp.STATE]["fracture_aperture_initial"].copy()

            data = param["layer_aperture"]["initial"]
            d[pp.STATE]["layer_aperture_initial"] = data(g, param, param["tol"])
            d[pp.STATE]["layer_aperture_star"] = d[pp.STATE]["layer_aperture_initial"].copy()
            d[pp.STATE]["layer_aperture"] = d[pp.STATE]["layer_aperture_initial"].copy()

            data = param["layer_porosity"]["initial"]
            d[pp.STATE]["layer_porosity_initial"] = data(g, param, param["tol"])
            d[pp.STATE]["layer_porosity_star"] = d[pp.STATE]["layer_porosity_initial"].copy()
            d[pp.STATE]["layer_porosity"] = d[pp.STATE]["layer_porosity_initial"] .copy()

            vol = d[pp.STATE]["porosity"] + d[pp.STATE]["fracture_aperture"] +\
                  d[pp.STATE]["layer_aperture"] * d[pp.STATE]["layer_porosity"]
            d[pp.STATE]["porosity_aperture_times_solute"] = vol * d[pp.STATE]["solute"]
            d[pp.STATE]["porosity_aperture_times_precipitate"] = vol * d[pp.STATE]["precipitate"]

            d[pp.STATE]["pressure"] = np.zeros(g.num_cells)
            d[pp.STATE]["P0_darcy_flux"] = np.zeros((3, g.num_cells))

        for e, d in self.gb.edges():
            d[pp.STATE][self.discr_flow.mortar] = np.zeros(d["mortar_grid"].num_cells)

        # set the old variables
        self.set_old_variables()

        # extract the initialized variables, useful for setting the data
        ###self.extract()

        # set now the data for each scheme
        self.discr_flow.set_data(param["flow"], param["time"])
        self.discr_temperature.set_data(param["temperature"], param["time"])
        self.discr_solute_advection_diffusion.set_data(param["solute_advection_diffusion"], param["time"])
        self.discr_solute_precipitate_reaction.set_data(param["solute_precipitate_reaction"], param["time"])
        self.discr_porosity.set_data(param["porosity"])
        self.discr_fracture_aperture.set_data(param["fracture_aperture"])
        self.discr_layer_aperture.set_data(param["layer_aperture"], param["time"])
        self.discr_layer_porosity.set_data(param["layer_porosity"])

    # ------------------------------------------------------------------------------#

    def compute_composite_variables(self):

        for g, d in self.gb:
            vol = d[pp.STATE]["porosity"] + d[pp.STATE]["fracture_aperture"] +\
                  d[pp.STATE]["layer_aperture"] * d[pp.STATE]["layer_porosity"]

            d[pp.STATE]["porosity_aperture_times_solute"] = vol * d[pp.STATE]["solute"]
            d[pp.STATE]["porosity_aperture_times_precipitate"] = vol * d[pp.STATE]["precipitate"]

    # ------------------------------------------------------------------------------#

    def vars_to_save(self):
        name = ["solute", "precipitate", "porosity", "fracture_aperture", "layer_aperture", "layer_porosity", "temperature"]
        name += ["porosity_aperture_times_solute", "porosity_aperture_times_precipitate", "fracture", "layer"]
        return name + [self.discr_flow.pressure, self.discr_flow.P0_flux]

    # ------------------------------------------------------------------------------#

    def one_step_splitting_scheme(self, time):

        # save the time of the current step
        self.time = time

        # POINT 1) extrapolate the precipitate to get a better estimate of porosity
        self.compute_precipitate_temperature_star()

        # POINT 2) compute the porosity and aperture star for both the fracture and layer
        self.compute_porosity_aperture_star()

        # -- DO THE FLOW PART -- #

        # POINT 3) update the data from the previous time step
        self.discr_flow.update_data()

        # POINT 4) solve the flow part
        self.compute_flow()

        # -- DO THE HEAT PART -- #

        # set the flux and update the data from the previous time step
        self.discr_temperature.set_flux(self.discr_flow.flux, self.discr_flow.mortar)
        self.discr_temperature.update_data()

        # POINT 5) solve the temperature part
        self.compute_temperature()

        # -- DO THE TRANSPORT PART -- #

        # set the flux and update the data from the previous time step
        self.discr_solute_advection_diffusion.set_flux(self.discr_flow.flux, self.discr_flow.mortar)
        self.discr_solute_advection_diffusion.update_data()

        # POINT 6) solve the advection and diffusion part to get the intermediate solute solution
        self.compute_solute_precipitate_advection_diffusion()

        # POINT 7) Since in the advection-diffusion step we have accounted for porosity changes using
        # phi_star and aperture_star, the new solute concentration accounts for the change in pore volume, thus, the
        # precipitate needs to be updated accordingly
        self.correct_precipitate_with_star()

        # POINT 8) solve the reaction part
        self.compute_solute_precipitate_rection()

        # -- DO THE POROSITY PART -- #

        # POINT 9) solve the porosity and aperture (fracture and layer) part with the true concentration of precipitate
        self.compute_porosity_aperture()

        # POINT 10) finally, we correct the concentrations to account for the difference between the extrapolated
        # and "true" new porosity to ensure mass conservation
        self.correct_precipitate_solute()

        # set the old variables
        self.set_old_variables()

        # compute composite variables
        self.compute_composite_variables()

    # ------------------------------------------------------------------------------#

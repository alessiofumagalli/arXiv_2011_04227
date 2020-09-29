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

        # -- layer aperture -- #
        self.discr_layer_aperture = LayerAperture(gb, "layer_aperture")

    # ------------------------------------------------------------------------------#

    def compute_precipitate_star(self):
        for g, d in self.gb:
            d[pp.STATE]["precipitate_star"] = 2*d[pp.STATE]["precipitate"] - d[pp.STATE]["precipitate_old"]

    # ------------------------------------------------------------------------------#

    def compute_porosity_aperture_star(self):
        for g, d in self.gb:
            d[pp.STATE]["porosity_star"] = self.discr_porosity.step(d[pp.STATE]["porosity_old"],
                                                                    d[pp.STATE]["precipitate_star"],
                                                                    d[pp.STATE]["precipitate_old"])

            d[pp.STATE]["fracture_aperture_star"] = self.discr_fracture_aperture.step(d[pp.STATE]["fracture_aperture_old"],
                                                                                      d[pp.STATE]["precipitate_star"],
                                                                                      d[pp.STATE]["precipitate_old"])

            d[pp.STATE]["layer_aperture_star"] = self.discr_layer_aperture.step(d[pp.STATE]["layer_aperture_old"],
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
                       d[pp.STATE]["layer_aperture_star"]

            vol_old = d[pp.STATE]["porosity_old"] + d[pp.STATE]["fracture_aperture_old"] + \
                      d[pp.STATE]["layer_aperture_old"]

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
        for g, d in self.gb:
            d[pp.STATE]["porosity"] = self.discr_porosity.step(d[pp.STATE]["porosity"],
                                                               d[pp.STATE]["precipitate_star_star"],
                                                               d[pp.STATE]["precipitate_old"])

            d[pp.STATE]["fracture_aperture"] = self.discr_fracture_aperture.step(d[pp.STATE]["fracture_aperture"],
                                                                                 d[pp.STATE]["precipitate_star_star"],
                                                                                 d[pp.STATE]["precipitate_old"])

            d[pp.STATE]["layer_aperture"] = self.discr_layer_aperture.step(d[pp.STATE]["layer_aperture"],
                                                                           d[pp.STATE]["precipitate_star_star"],
                                                                           d[pp.STATE]["precipitate_old"])

    # ------------------------------------------------------------------------------#

    def correct_precipitate_solute(self):
        for g, d in self.gb:
            vol_star = d[pp.STATE]["porosity_star"] + d[pp.STATE]["fracture_aperture_star"] +\
                       d[pp.STATE]["layer_aperture_star"]

            vol = d[pp.STATE]["porosity"] + d[pp.STATE]["fracture_aperture"] + \
                      d[pp.STATE]["layer_aperture"]

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

    # ------------------------------------------------------------------------------#

    def set_data(self, param):

        for g, d in self.gb:
            data = param["temperature"]["initial"]
            d[pp.STATE]["temperature"] = data(g, param, param["tol"])

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

            vol = d[pp.STATE]["porosity"] + d[pp.STATE]["fracture_aperture"] + d[pp.STATE]["layer_aperture"]
            d[pp.STATE]["porosity_aperture_times_solute"] = vol * d[pp.STATE]["solute"]
            d[pp.STATE]["porosity_aperture_times_precipitate"] = vol * d[pp.STATE]["precipitate"]

            d[pp.STATE]["pressure"] = np.zeros(g.num_cells)
            d[pp.STATE]["P0_darcy_flux"] = np.zeros((3, g.num_cells))

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
        self.discr_layer_aperture.set_data(param["layer_aperture"])

    # ------------------------------------------------------------------------------#

    def compute_composite_variables(self):

        for g, d in self.gb:
            vol = d[pp.STATE]["porosity"] + d[pp.STATE]["fracture_aperture"] +\
                  d[pp.STATE]["layer_aperture"]

            d[pp.STATE]["porosity_aperture_times_solute"] = vol * d[pp.STATE]["solute"]
            d[pp.STATE]["porosity_aperture_times_precipitate"] = vol * d[pp.STATE]["precipitate"]

    # ------------------------------------------------------------------------------#

    def vars_to_save(self):
        name = ["solute", "precipitate", "porosity", "fracture_aperture", "layer_aperture", "temperature"]
        name += ["porosity_aperture_times_solute", "porosity_aperture_times_precipitate", "fracture", "layer"]
        return name + [self.discr_flow.pressure, self.discr_flow.P0_flux]

    # ------------------------------------------------------------------------------#

    def one_step_splitting_scheme(self):

        # POINT 1) extrapolate the precipitate to get a better estimate of porosity
        self.compute_precipitate_star()

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

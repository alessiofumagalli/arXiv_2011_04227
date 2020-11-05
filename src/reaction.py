import numpy as np

class Reaction(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, model="reaction"):

        self.model = model
        self.data = None

    # ------------------------------------------------------------------------------#

    def set_data(self, data, data_time):

        self.data = data
        self.data_time = data_time

        # bisection parameters
        self.tol = data["tol_reaction"]
        # this tolerance should be really small
        self.tol_consider_zero = data["tol_consider_zero"]
        self.max_iter = data["max_iter"]

        # dimensionless parameter
        self.adim = data["length"]/data["gamma_eq"]/data["velocity"]

        self.reaction = data["reaction"]

        self.time_step = data_time["step"]

    # ------------------------------------------------------------------------------#

    def step(self, solute, precipitate, temperature):

        # ensure that are array
        solute = np.atleast_1d(solute)
        precipitate = np.atleast_1d(precipitate)
        temperature = np.atleast_1d(temperature)

        # create the solution
        uw_0 = np.vstack((solute, precipitate))

        # solution at the end of the procedure
        uw_n = np.zeros(uw_0.shape)

        # compute the reaction rate with the solute and precipitate
        reaction_0 = self.reaction_fct(uw_0, temperature)

        # update the solute and precipitate
        uw_1 = uw_0 + 0.5 * self.time_step * reaction_0

        # cut values too small
        too_small = np.abs(uw_1) - self.tol_consider_zero < 0
        uw_1[too_small] = 0

        # we need to check now if the precipitate went negative
        neg_1 = np.where(uw_1[1, :] < 0)[0]
        pos_1 = np.where(uw_1[1, :] >= 0)[0]

        # for them compute the time step such that they become null, only if it happens
        if neg_1.size > 0:
            time_step_corrected = np.tile(uw_0[1, neg_1] / np.abs(reaction_0[1, neg_1]), (2, 1))

            # correct the step
            uw_corrected = uw_0[:, neg_1] + 0.5 * time_step_corrected * reaction_0[:, neg_1]
            uw_eta = uw_0[:, neg_1] + time_step_corrected * self.reaction_fct(uw_corrected, temperature[neg_1])

            # then finish with the last part of the time_step
            delta = self.time_step - time_step_corrected
            uw_corrected = uw_eta + 0.5 * delta * self.reaction_fct(uw_eta, temperature[neg_1])
            uw_n[:, neg_1] = uw_eta + delta * self.reaction_fct(uw_corrected, temperature[neg_1])

        # consider now the positive and progress with the scheme, only if it ahppens
        if pos_1.size > 0:
            uw_n[:, pos_1] = uw_0[:, pos_1] + self.time_step * self.reaction_fct(uw_1[:, pos_1], temperature[pos_1])

        # cut values too small
        too_small = np.abs(uw_n) - self.tol_consider_zero < 0
        uw_n[too_small] = 0

        # if during the second step something goes wrong we still need to work on it
        neg_n = pos_1[np.where(uw_n[1, pos_1] < 0)[0]]

        for i in neg_n:
            # initialization of the bisection algorithm
            if uw_n[1, i] < 0:
                err = np.abs(uw_n[1, i])
                it = 0
                time_step_a, time_step_b = 0, self.time_step

                uw_bisec = uw_0[:, i]
                time_step_mid = self.time_step

                while err > self.tol and it < self.max_iter:
                    time_step_mid = 0.5*(time_step_a + time_step_b)
                    # re-do the step
                    uw_bisec = uw_0[:, i, np.newaxis] + 0.5 * time_step_mid * reaction_0[:, i, np.newaxis]
                    uw_bisec = uw_0[:, i, np.newaxis] + time_step_mid * self.reaction_fct(uw_bisec, temperature[i])

                    if uw_bisec[1, :] > 0:
                        time_step_a = time_step_mid
                    else:
                        time_step_b = time_step_mid

                    it += 1
                    err = np.abs(uw_bisec[1, :])

                # we are at convergence, we want to avoid negative numbers
                uw_bisec[1, :] = 0

                # then finish with the last part of the time_step
                delta = self.time_step - time_step_mid
                uw_corrected = uw_bisec + 0.5 * delta * self.reaction_fct(uw_bisec, temperature[i])
                uw_n[:, i, np.newaxis] = uw_bisec + delta * self.reaction_fct(uw_corrected, temperature[i])

        return uw_n[0, :], uw_n[1, :]

    # ------------------------------------------------------------------------------#

    def reaction_fct(self, uw, temperature):
        # set the reaction rate, we suppose given as a function with
        # (solute, precipitate) as argument

        sign = np.ones(uw.shape)
        sign[1, :] = -1

        val = self.reaction(uw[0, :], uw[1, :], temperature)
        return self.adim * np.einsum("ij,j->ij", sign, val)

    # ------------------------------------------------------------------------------#

import numpy as np
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from scheme import Scheme

from data import create_gb, get_param

def main():

    mesh_size = np.power(2., 3)
    gb = create_gb(mesh_size)

    param = get_param()

    scheme = Scheme(gb)
    scheme.set_data(param)

    # exporter
    folder = "./case4/"
    save = pp.Exporter(gb, "case2", folder_name=folder+"caseYY/")
    vars_to_save = scheme.vars_to_save()

    # post process
    save.write_vtk(vars_to_save, time_step=0)

    for i in np.arange(param["time"]["num_steps"]):
        time = (i+1)*param["time"]["step"]

        print("processing", i, "step at time", time)

        # do one step of the splitting scheme
        scheme.one_step_splitting_scheme(time)

        # post process
        save.write_vtk(vars_to_save, time_step=time)

    time = np.arange(param["time"]["num_steps"]+1)*param["time"]["step"]
    save.write_pvd(time)

if __name__ == "__main__":
    main()

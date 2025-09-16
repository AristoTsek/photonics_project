import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import tidy3d as td
from tidy3d.plugins import waveguide
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.dispersion import AdvancedFastFitterParam, FastDispersionFitter
from config import n_sio2, n_si, plot_refractive_si, plot_refractive_sio2, get_config, process_data_from_directory, read_hfss_csv
from config import plot_ex1, group_index_calc, plot_ex2, plot_ex3
from tidy3d import GridSpec, AutoGrid
from scipy import interpolate
from scipy.integrate import dblquad

def main():
    # Configuring the fonts of the plots globally
    config = get_config()
    plt.rcParams['axes.labelsize'] = config["font"]
    plt.rcParams['xtick.labelsize'] = config["font"]
    plt.rcParams['ytick.labelsize'] = config["font"]
    plt.rcParams['axes.titlesize'] = config["font"]

    # Calculating the starting and ending frequencies in Hz
    freq_end = (config["c0"] / config["wavelength_end"]) * 1e12
    freq_start = (config["c0"] / config["wavelength_start"]) * 1e12

    if config["verbose"]:
        print("Starting wavelength:", config["wavelength_start"] * 1e3, " nm")
        print("Starting frequency:", freq_start * 1e-12, " THz")
        print("Ending wavelength:", config["wavelength_end"] * 1e3, " nm")
        print("Ending frequency:", freq_end * 1e-12, " THz")

    # The following code creates a dispersive material for describing silicon and silicon dioxide on tidy3d
    fitter_si = FastDispersionFitter.from_file("si.csv", skiprows=0, delimiter=",")
    fitter_sio2 = FastDispersionFitter.from_file("sio2.csv", skiprows=0, delimiter=",")

    advanced_param_si = AdvancedFastFitterParam(weights=(1, 1))
    silicon_material, rms_error1 = fitter_si.fit(max_num_poles=3, advanced_param=advanced_param_si, tolerance_rms=1e-12)
    fitter_si.plot(silicon_material)
    if config["plot_live"]:
        plt.show()
    else:
        plt.savefig("./silicon_tidy3d_fit.png", format="png")

    advanced_param_sio2 = AdvancedFastFitterParam(weights=(1, 1))
    silicon_dioxide_material, rms_error2 = fitter_sio2.fit(max_num_poles=3, advanced_param=advanced_param_sio2, tolerance_rms=1e-12)
    fitter_sio2.plot(silicon_dioxide_material)
    if config["plot_live"]:
        plt.show()
    else:
        plt.savefig("./silicon_dioxide_tidy3d_fit.png", format="png")

    # Creating two subplots for the refractive index of silicon and silicon dioxide
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(18, 8))

    plot_refractive_sio2(config["wavelength_start"], config["wavelength_end"], 0.01, ax1)
    plot_refractive_si(config["wavelength_start"], config["wavelength_end"], 0.01, ax2)

    if config["plot_live"]:
        plt.show()
    else:
        plt.savefig("./materials_refractive_index.png", format="png")

    # Tidy3d strip waveguide creation and cross section plot
    # Strip created at 1.31 um wavelength
    strip_1p31 = waveguide.RectangularDielectric(
        wavelength=config["wavelength_no1"],
        core_width=config["waveguide_width"],
        core_thickness=config["waveguide_height"],
        core_medium=silicon_material,
        clad_medium=td.Medium(permittivity=1.0), # Cladding with air - constant permittivity
        box_medium=silicon_dioxide_material,
        mode_spec=td.ModeSpec(num_modes=2, group_index_step=True), # Two modes for Quasi-TE and Quasi-TM
        propagation_axis=2,
        normal_axis=1,
        grid_resolution=64
    )

    # Strip created at 1.55 um wavelength
    strip_1p55 = waveguide.RectangularDielectric(
        wavelength=config["wavelength_no2"],
        core_width=config["waveguide_width"],
        core_thickness=config["waveguide_height"],
        core_medium=silicon_material,
        clad_medium=td.Medium(permittivity=1.0), # Cladding with air - constant permittivity
        box_medium=silicon_dioxide_material,
        mode_spec=td.ModeSpec(num_modes=2, group_index_step=True), # Two modes for Quasi-TE and Quasi-TM
        propagation_axis=2,
        normal_axis=1,
    )

    ax = strip_1p31.plot_structures(z=0) # strip_1p31 and strip_1p55 will have the same cross section
    ax.set_xlabel('x [μm]')
    ax.set_ylabel('y [μm]')
    ax.set_title('cross section at z = 0.00 [μm]')

    if config["plot_live"]:
        plt.show()
    else:
        plt.savefig("./cross_section_tidy3d.png", format="png")

    # |------------|
    # | Exercise 1 |
    # |------------|
    # Find profiles for Quasi-TE and Quasi-TM (first two modes) at 1.55 and 1.31 um
    # Plot |E|, |Ex|, |Ey|, |Ez| for each case
    # Max total 16 plots for each tool (HFSS, online, tidy3d)

    
    # Uncomment the line below to run exercise 1
    plot_ex1(strip_1p31, strip_1p55, config)

    # |------------|
    # | Exercise 2 |
    # |------------|
    # Strip waveguide to calculate effective index, for 150 wavelength values
    strip_neff = waveguide.RectangularDielectric(
        wavelength=np.linspace(1.3, 1.9, 150),
        core_width=config["waveguide_width"],
        core_thickness=config['waveguide_height'],
        core_medium=silicon_material,
        clad_medium=td.Medium(permittivity=1.0),
        box_medium=silicon_dioxide_material,
        mode_spec=td.ModeSpec(num_modes=2, group_index_step=True),
        propagation_axis=2,
        normal_axis=1
    )

    # Uncomment the line below to run exercise 2
    n_on_te, hfss_n_eff_0, hfss_n_eff_1, lambdas, hfss_x = plot_ex2(strip_neff, config)

    # |------------|
    # | Exercise 3 |
    # |------------|
    # To run exercise 3 you need to run exercise 2 to get the necessary outputs

    # Uncomment the line below to run exercise 3
    plot_ex3(strip_neff, config, n_on_te, lambdas, hfss_n_eff_0, hfss_x, hfss_n_eff_1)

    # |------------|
    # | Exercise 4 |
    # |------------|
    # The calculations of the integrals may take some time!

    # Effective area calculation for tidy3d (instant)
    Aeff_tidy3d = np.asarray(strip_neff.mode_solver.data.mode_area).squeeze()

    TE_Hfss, x_data_hfss_te, y_data_hfss_te = read_hfss_csv("TE_E.csv", 75)
    TM_Hfss, x_data_hfss_tm, y_data_hfss_tm = read_hfss_csv("TM_E.csv", 75)

    power_of_two = []
    power_of_four = []
    power_of_two_hfss = []
    power_of_four_hfss = []

    nominator = np.zeros(shape=(55)) # 55 frequency points for online simulator
    denominator = np.zeros(shape=(55))

    nominator_hfss = np.zeros(shape=(75)) # 75 frequency points for hfss
    denominator_hfss = np.zeros(shape=(75))

    for z in TE_Hfss:
        z_this = z * 1e-6 # turn meters to micrometers
        power_of_two_hfss.append(z_this ** 2)
        power_of_four_hfss.append(z_this ** 4)

    freqs_hfss = np.array(
        [157.78550, 158.76961, 159.75372, 160.73783, 161.72194, 162.70605, 163.69016, 164.67427, 165.65838,
         166.64249, 167.62659, 168.61070, 169.59481, 170.57892, 171.56303, 172.54714, 173.53125, 174.51536,
         175.49947, 176.48358, 177.46769, 178.45180, 179.43591, 180.42002, 181.40413, 182.38824, 183.37235,
         184.35646, 185.34056, 186.32467, 187.30878, 188.29289, 189.27700, 190.26111, 191.24522, 192.22933,
         193.21344, 194.19755, 195.18166, 196.16577, 197.14988, 198.13399, 199.11810, 200.10221, 201.08632,
         202.07043, 203.05454, 204.03864, 205.02275, 206.00686, 206.99097, 207.97508, 208.95919, 209.94330,
         210.92741, 211.91152, 212.89563, 213.87974, 214.86385, 215.84796, 216.83207, 217.81618, 218.80029,
         219.78440, 220.76851, 221.75261, 222.73672, 223.72083, 224.70494, 225.68905, 226.67310, 227.65727,
         228.64138, 229.62549, 230.60960])  # THz (75 frequencies taken from simulations)

    lambdas_hfss = (config["c0"] / freqs_hfss) * 1e3 # turn frequencies to wavelength

    # Calculating the nominator for each wavelength
    for num in range(0, len(power_of_two_hfss)):
        f = interpolate.RectBivariateSpline(x_data_hfss_te, y_data_hfss_te, power_of_two_hfss[num]) # interpolation
        xmin = x_data_hfss_te.min()
        xmax = x_data_hfss_te.max()
        ymin = y_data_hfss_te.min()
        ymax = y_data_hfss_te.max()
        integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
        nominator_hfss[num] = integral ** 2

    print("HFSS NOMINATOR CALCULATION DONE!")

    # Calculating the denominator for each wavelength
    for num in range(0, len(power_of_four_hfss)):
        f = interpolate.RectBivariateSpline(x_data_hfss_te, y_data_hfss_te, power_of_four_hfss[num])
        xmin = x_data_hfss_te.min()
        xmax = x_data_hfss_te.max()
        ymin = y_data_hfss_te.min()
        ymax = y_data_hfss_te.max()
        integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
        denominator_hfss[num] = integral

    print("HFSS DENOMINATOR CALCULATION DONE!")

    file_path_cons_online = 'all_plots_online'

    z_data, x_data, y_data = process_data_from_directory(file_path_cons_online, "te", ".xyz")

    for z in z_data:
        power_of_two.append(z ** 2)
        power_of_four.append(z ** 4)

    # Calculate the nominators for online simulator for each wavelength
    for num in range(0, len(power_of_two)):
        f = interpolate.RectBivariateSpline(y_data, x_data, power_of_two[num])
        xmin = x_data.min()
        xmax = x_data.max()
        ymin = y_data.min()
        ymax = y_data.max()
        integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
        nominator[num] = integral ** 2

    print("Finished calculating nominators for TE online.")

    # Calculate the denominators for online simulator for each wavelength
    for num in range(0, len(power_of_four)):
        f = interpolate.RectBivariateSpline(y_data, x_data, power_of_four[num])
        xmin = x_data.min()
        xmax = x_data.max()
        ymin = y_data.min()
        ymax = y_data.max()
        integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
        denominator[num] = integral

    print("Finished calculating denominators for TE online.")

    # Calculate effective area for each tool
    Aeff = nominator / denominator
    Aeff_hfss = nominator_hfss / denominator_hfss

    print(nominator_hfss)
    print(denominator_hfss)

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas_hfss, Aeff_hfss, linestyle='--', color='black', label='quasi-TE HFSS')
    plt.scatter(np.arange(1300, 1850, 10), Aeff, marker='o', label='quasi-TE οnline')
    plt.plot(np.linspace(1300, 1900, 150), Aeff_tidy3d[:, 0], linestyle='-', color='green', label='quasi-TE Tidy3d')

    plt.xlabel("Wavelength [nm]", fontsize=config["font"])
    plt.xticks(fontsize=config["font"])
    plt.yticks(fontsize=config["font"])
    plt.ylabel("A$_e$$_f$$_f$ [μm$^2$]", fontsize=config["font"])
    plt.grid(True)
    plt.legend(fontsize=config["font"])
    plt.show()

    ############## TM Calculations ################
    power_of_two_hfss = []
    power_of_four_hfss = []
    nominator_hfss = np.zeros(shape=(75))
    denominator_hfss = np.zeros(shape=(75))

    for z in TM_Hfss:
        z_this = z * 1e-6
        power_of_two_hfss.append(z_this ** 2)
        power_of_four_hfss.append(z_this ** 4)

    for num in range(0, len(power_of_two_hfss)):
        f = interpolate.RectBivariateSpline(x_data_hfss_tm, y_data_hfss_tm, power_of_two_hfss[num])
        xmin = x_data_hfss_tm.min()
        xmax = x_data_hfss_tm.max()
        ymin = y_data_hfss_tm.min()
        ymax = y_data_hfss_tm.max()
        integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
        nominator_hfss[num] = integral ** 2

    print("HFSS NOMINATOR CALCULATION DONE!")

    for num in range(0, len(power_of_four_hfss)):
        f = interpolate.RectBivariateSpline(x_data_hfss_tm, y_data_hfss_tm, power_of_four_hfss[num])
        xmin = x_data_hfss_tm.min()
        xmax = x_data_hfss_tm.max()
        ymin = y_data_hfss_tm.min()
        ymax = y_data_hfss_tm.max()
        integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
        denominator_hfss[num] = integral

    print("HFSS DENOMINATOR CALCULATION DONE!")

    Aeff_hfss = nominator_hfss / denominator_hfss

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas_hfss, Aeff_hfss, linestyle='--', color='black', label='quasi-TM HFSS')
    plt.plot(np.linspace(1300, 1900, 150), Aeff_tidy3d[:, 1], linestyle='-', color='green', label='quasi-TM Tidy3d')

    plt.xlabel("Wavelength [nm]", fontsize=config["font"])
    plt.xticks(fontsize=config["font"])
    plt.yticks(fontsize=config["font"])
    plt.ylabel("A$_e$$_f$$_f$ [μm$^2$]", fontsize=config["font"])
    plt.grid(True)
    plt.legend(fontsize=config["font"])
    plt.show()

if __name__ == "__main__":
    main()



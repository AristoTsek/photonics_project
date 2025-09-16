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

    #plot_ex1(strip_1p31, strip_1p55, config)

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

    #n_on_te, hfss_n_eff_0, hfss_n_eff_1, lambdas, hfss_x = plot_ex2(strip_neff, config)

    # |------------|
    # | Exercise 3 |
    # |------------|
    # To run exercise 3 you need to run exercise 2 to get the necessary outputs

    #plot_ex3(strip_neff, config, n_on_te, lambdas, hfss_n_eff_0, hfss_x, hfss_n_eff_1)

    # |------------|
    # | Exercise 4 |
    # |------------|

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














strip3 = waveguide.RectangularDielectric(
    wavelength=np.linspace(1.3, 1.9, 60),
    core_width=width,
    core_thickness=height,
    core_medium=silicon_material,
    clad_medium=td.Medium(permittivity=1.0),
    box_medium=silicon_dioxide_material,
    mode_spec=td.ModeSpec(num_modes=2, group_index_step=True),
    propagation_axis=2,
    normal_axis=1
)

n_eff = strip3.mode_solver.data.n_eff
n_g = strip3.mode_solver.data.n_group
n_0 = n_eff[:, 1]
waves = np.linspace(1.3, 1.9, 60)

dneff_dlambda = np.gradient(n_0, waves)  # Central difference

# Calculate the group index
ng = n_0 - waves * dneff_dlambda
print(ng)
print(n_g[:, 1])


n_eff.plot.line(x="f")
plt.show()

#|------------|
#| Exercise 2 |
#|------------|
# Values are taken from online simulator


# Quasi-TM 1.55 - 1.86
beta_on_tm = [10.30912651, 10.10525436, 9.904543897, 9.706837611,
              9.511988296, 9.319858108, 9.130317702, 8.94324546,
              8.758526791, 8.576053506, 8.395723241, 8.217438933,
              8.041108347, 7.866643623, 7.693960877, 7.522979809,
              7.353623345, 7.185817298, 7.019490038, 6.854572174,
              6.690996247, 6.528696422, 6.367608183, 6.207668015,
              6.048813089, 5.89098092, 5.734109011, 5.578134468,
              5.422993579, 5.268621348, 5.114950978, 4.961913271]

n_on_tm = [2.543160086, 2.508949844, 2.474880679, 2.440928076,
           2.40706913, 2.373282379, 2.339547663, 2.30584599,
           2.272159418, 2.238470945, 2.20476441, 2.171024403,
           2.137236176, 2.103385567, 2.06945892, 2.03544302,
           2.001325014, 1.96709235, 1.93273271, 1.898233937,
           1.863583972, 1.828770781, 1.793782283, 1.758606268,
           1.72323032, 1.68764172, 1.651827346, 1.615773566,
           1.579466109, 1.542889921, 1.506029004, 1.468866225]

# Quasi-TE 1.3 - 1.85
beta_on_te = [13.03720983, 12.8668847, 12.69856351, 12.53217117,
           12.36763572, 12.20488784, 12.04386044, 11.88448836,
           11.72670805, 11.5704574, 11.41567548, 11.2623024,
           11.11027913, 10.95954734, 10.81004924, 10.66172745,
           10.51452484, 10.36838442, 10.22324912, 10.07906171,
           9.935764596, 9.79329965, 9.651608056, 9.510630091,
           9.370304927, 9.230570396, 9.091362743, 8.952616343,
           8.814263395, 8.676233573, 8.538453636, 8.400846986,
           8.263333161, 8.125827257, 7.988239264, 7.850473285,
           7.712426632, 7.573988764, 7.435040024, 7.295450136,
           7.155076394, 7.013761478, 6.871330771, 6.727589058,
           6.582316398, 6.435262926, 6.286142204, 6.134622614,
           5.980316023, 5.82276261, 5.661410124, 5.495584898,
           5.324450227, 5.146944721, 4.961687525]




# size of simulation domain
Lx, Ly, Lz = 1.5, 2, 1
dl = 0.01

# run_time in ps
run_time = 1e-12

# automatic grid specification
grid_spec = td.GridSpec.auto(min_steps_per_wvl=20, wavelength=1.55)

sim = td.Simulation(
    size=(Lx, Ly, Lz),
    grid_spec=grid_spec,
    structures=strip.structures,
    run_time=run_time,
)

ax = sim.plot(z=0)
plt.show()

plane = td.Box(center=(-0.6, -1, 0), size=(0.9, 1, 0))

mode_spec = td.ModeSpec(
    num_modes=2,
    target_neff=2.0,
    group_index_step=True
)

num_freqs = 60
f0_ind = num_freqs // 2
freqs = np.linspace(freq_start, freq_end, num_freqs)

mode_solver = ModeSolver(
    simulation=sim,
    plane=plane,
    mode_spec=mode_spec,
    freqs=freqs,
)
mode_data = mode_solver.solve()

mode_data.to_dataframe()

fig, ax = plt.subplots(1)
n_eff = mode_data.n_eff
print(n_eff)


file_path = 'Gamma.csv'

# Function to plot HFSS results compared to the results of the online simulator
def plot_hfss_online(hfss_x, hfss_y, online_x1, online_y1, online_x2, online_y2,
                     x_axis_name, y_axis_name, titlename):
    plt.figure(figsize=(10, 7))
    plt.plot(hfss_x, hfss_y, linestyle='--', color='black', label='quasi-TE HFSS')
    plt.scatter(online_x1, online_y1, marker='o', color='blue', label='quasi-TE online')
    plt.scatter(online_x2, online_y2, marker='o', color='red', label='quasi-TM online')

    plt.xlabel(x_axis_name, fontsize=22)  # Common x label
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylabel(y_axis_name, fontsize=22)
    plt.title(titlename, fontsize=22)
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.show()

df = pd.read_csv(file_path)

y_column = df.columns[0] # First column is y
x1_column = df.columns[1] # Second column is mode 1
x2_column = df.columns[2] # Third column is mode 2
hfss_x = 299792458 / df[y_column] * 1e-6
hfss_y = [df[x2_column][x] * 1e-6 for x in range(0, len(df[x2_column]))]
online_x1 = np.arange(1.3, 1.85, 0.01)
online_x2 = np.arange(1.55, 1.86, 0.01)
online_y1 = beta_on_te
online_y2 = beta_on_tm
x_axis = "Wavelength [μm]"
y_axis = "Propagation constant [μm$^-$$^1$]"
#plot_hfss_online(hfss_x, hfss_y, online_x1, online_y1, online_x2, online_y2, x_axis, y_axis,
#                 "Propagation Constant vs Wavelength")

# Calculating wave number
k = 2 * math.pi / (299792458 / df[y_column] * 1e-6)

neff = []

# Calculating effective indexes
for x in range (0, len(df[x2_column])):
    neff.append((df[x2_column][x] * 1e-6) / k[x])

hfss_y = neff
online_y1 = n_on_te
online_y2 = n_on_tm
x_axis = "Wavelength [μm]"
y_axis = "N$_e$$_f$$_f$"
plot_hfss_online(hfss_x, hfss_y, online_x1, online_y1, online_x2, online_y2, x_axis, y_axis,
                 "Effective Index vs Wavelength")

########---------------------Exercise 3---------------------########

# Calculating HFSS Group Index
lambdas = 299792458 / df[y_column] * 1e-6
dneffdlambdas = np.gradient(neff, lambdas)

ng_hfss = neff - lambdas * dneffdlambdas

# Calculating Online TE Group Index
lambdas2 = np.arange(1.3, 1.85, 0.01)
diff_lambdas = 0.01
dneffdlambdas2 = np.gradient(n_on_te, lambdas2)

ng_te = n_on_te - lambdas2 * dneffdlambdas2

# Calculating Online TM Group Index
lambdas3 = np.arange(1.55, 1.86, 0.01)
dneffdlambdas3 = np.gradient(n_on_tm, lambdas3)

ng_tm = n_on_tm - lambdas3 * dneffdlambdas3

hfss_y = ng_hfss
online_y1 = ng_te
online_y2 = ng_tm
x_axis = "Wavelength [μm]"
y_axis = "n$_g$"
plot_hfss_online(hfss_x, hfss_y, online_x1, online_y1, online_x2, online_y2, x_axis, y_axis,
                 "Group Index vs Wavelength")


exit()


# Define parameters
wave_start = 1.31
wave_stop = 1.55
inter = 0.1

# Define the n_si equation
def n_si(x):
    denom1 = 1 - (0.301516485 / x)**2
    denom2 = 1 - (1.13475115 / x)**2
    denom3 = 1 - (1104 / x)**2
    term1 = 1
    term2 = 10.6684293 / denom1
    term3 = 0.0030434748 / denom2
    term4 = 1.54133408 / denom3
    return math.sqrt(term1 + term2 + term3 + term4)

# Define the n_sio2 equation
def n_sio2(x):
    denom1 = 1 - (0.0684043 / x)**2
    denom2 = 1 - (0.1162414 / x)**2
    denom3 = 1 - (9.896161 / x)**2
    term1 = 1
    term2 = 0.6961663 / denom1
    term3 = 0.4079426 / denom2
    term4 = 0.8974794 / denom3
    return math.sqrt(term1 + term2 + term3 + term4)

# Plot the refractive index of silicon for a specific range
def plot_refractive_si(lower_b, upper_b, step):

    plt.figure(figsize=(10, 7))
    all_values = np.arange(lower_b, upper_b, step)
    n_si_values = np.zeros(shape=int((upper_b - lower_b) / step) + 1)
    for i, x  in enumerate(all_values):
        n_si_values[i] = n_si(x)
    plt.plot( all_values, n_si_values, linestyle='-', color='black')
    plt.xlabel("Wavelength [μm]", fontsize=22)
    plt.ylabel("Refractive Index Si", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True)
    plt.title("Refractive Index of Silicon", fontsize=22)
    plt.savefig("ref_index_silicon.svg")

# Plot the refractive index of silicon dioxide for a specific range
def plot_refractive_sio2(lower_b, upper_b, step):

    plt.figure(figsize=(11, 7))
    all_values = np.arange(lower_b, upper_b, step)
    n_sio2_values = np.zeros(shape=int((upper_b - lower_b) / step) + 1)
    for i, x in enumerate(all_values):
        n_sio2_values[i] = n_sio2(x)
    plt.plot(all_values, n_sio2_values, linestyle='-', color='black')
    plt.xlabel("Wavelength [μm]", fontsize=22)
    plt.ylabel(r"Refractive Index SiO$_2$", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True)
    plt.title("Refractive Index of Silicon Dioxide", fontsize=22)
    plt.savefig("ref_index_silicon_dio.svg")


# Testing
wavelength = 1.3 # wavelength parameter

print("For wavelength of ", wavelength, " um, n_si = ", n_si(wavelength))
print("For wavelength of ", wavelength, " um, n_sio2 = ", n_sio2(wavelength))

plot_refractive_sio2(1.3, 1.9, 0.01)
plot_refractive_si(1.3, 1.9, 0.01)



exit(1)
####################################################################

########---------------------Exercise 1---------------------########

def process_csv_data(directory):

    z_arrays = []
    x_coords = None
    y_coords = None
    first_file_processed = False

    filepath = os.path.join(directory)
    try:
        data = pd.read_csv(filepath)

        for it in range(2, len(data.columns)):
            array = data[data.columns[it]].to_numpy()
            z_arrays.append(array.reshape(75, 100))  # Extract third column (values)

        if not first_file_processed:
            x_coords = data[data.columns[1]].to_numpy()
            y_coords = data[data.columns[0]].to_numpy()
            x_coords_uniq = np.zeros(shape=(75))
            y_coords_uniq = np.zeros(shape=(100))
            # Get only the unique values
            i = 0
            for x in x_coords:
                if x not in x_coords_uniq:
                    x_coords_uniq[i] = x
                    i += 1
            i = 0
            for y in y_coords:
                if y not in y_coords_uniq:
                    print(y)
                    y_coords_uniq[i] = y
                    i += 1

            first_file_processed = True
    except (FileNotFoundError, ValueError) as e:
        print(f"Error processing file '{directory}': {e}")
        return None, None, None  # Return None if any error occurs


    if not z_arrays:
        print("No suitable files found in the directory.")
        return None, None, None

    return z_arrays, np.sort(x_coords_uniq), np.sort(y_coords_uniq)


def plot_profile(z_data, titlename):

    plt.figure(figsize=(9, 7))
    plt.imshow(z_data)
    plt.xlabel("X", fontsize=22)  # Common x label
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)
    plt.ylabel("Y", fontsize=22)
    plt.title(titlename, fontsize=22)
    plt.grid(True)
    plt.show()

# Plotting Magnitude of E (|E|) in V/um
file_path = 'all_data.csv'
z_data_hfss, x_data_hfss, y_data_hfss = process_csv_data(file_path)

directory_path = "all_plots"  # Replace with your directory path
z_data, x_data, y_data = process_data_from_directory(directory_path, "te", ".xyz")
z_data_tm, x_data_tm, y_data_tm = process_data_from_directory(directory_path, "tm",".xyz")

plot_profile(z_data[1], "Profile of |E| for TE at 1.31 μm, Online")
plot_profile(z_data[25], "Profile of |E| for TE at 1.55 μm, Online")
plot_profile(z_data_tm[0], "Profile of |E| for TM at 1.55 μm, Online")

plot_profile(z_data_hfss[72] * 1e-6, "Profile of |E| at 1.31 μm, HFSS")
plot_profile(z_data_hfss[36] * 1e-6, "Profile of |E| at 1.55 μm, HFSS")

# Plotting the magnitudes of Ex, Ey, Ez for each mode
file_path_cons = 'cons_hfss'
file_path_cons_online = 'cons_online'
array_titles = ["Ey", "Ex", "Ez"]
array_titles2 = ["Ex for TE", "Ey for TE", "Ez for TE", "Ex for TM", "Ey for TM", "Ez for TM"]

for i, filename in enumerate(os.listdir(file_path_cons)):
    z_hfss, x_hfss, y_hfss = process_csv_data(file_path_cons + "/" + filename)
    plot_profile(z_hfss[0], "Profile of " + array_titles[i] + " at 1.55 μm, HFSS")
    plot_profile(z_hfss[1], "Profile of " + array_titles[i] + " at 1.31 μm, HFSS")

z, x, y = process_data_from_directory(file_path_cons_online, "", ".xyz")

for i in range(0, 6):
    if i < 3:
        plot_profile(z[2*i], "Profile of " + array_titles2[i] + " at 1.31 μm, Online")
        plot_profile(z[2*i + 1], "Profile of " + array_titles2[i] + " at 1.55 μm, Online")
    else:
        plot_profile(z[3 + i], "Profile of " + array_titles2[i] + " at 1.55 μm, Online")

########---------------------Exercise 2---------------------########




########---------------------Exercise 4---------------------########

power_of_two = []
power_of_four = []
power_of_two_tm = []
power_of_four_tm = []
power_of_two_hfss = []
power_of_four_hfss = []

for z in z_data:
    power_of_two.append(z ** 2)
    power_of_four.append(z ** 4)

for z in z_data_tm:
    power_of_two_tm.append(z ** 2)
    power_of_four_tm.append(z ** 4)

nominator = np.zeros(shape=(55))
denominator = np.zeros(shape=(55))
nominator_tm = np.zeros(shape=(32))
denominator_tm = np.zeros(shape=(32))
nominator_hfss = np.zeros(shape=(75))
denominator_hfss = np.zeros(shape=(75))

print(z_data_tm[0])
print(z_data[0])
print(z_data_hfss[0])

for z in z_data_hfss:
    z_this = z.T * 1e-6
    power_of_two_hfss.append(z_this ** 2)
    power_of_four_hfss.append(z_this ** 4)

freqs = np.array([157.7855, 158.763, 159.74051, 160.71801, 161.69552, 162.67302, 163.65053, 164.62803, 165.60554,
         166.58304, 167.56055, 168.53805, 169.51556, 170.49306, 171.47057, 172.44807, 173.42558, 174.40308,
         175.38058, 176.35809, 177.33559, 179.29060, 180.26811, 181.24561, 182.22312, 183.20062, 184.17813,
         185.15563, 186.13314, 187.11064, 188.08815, 189.06565, 190.04316, 191.02066, 191.99816, 192.97567,
         193.95317, 194.93068, 195.90818, 196.88569, 197.86319, 198.84070, 199.81820, 200.79571, 201.77321,
         202.75072, 203.72822, 204.70573, 205.68323, 206.66073, 207.63824, 208.61574, 209.59325, 210.57075,
         211.54826, 212.52576, 213.50327, 214.48077, 215.45828, 216.43578, 217.41329, 218.39079, 219.36830,
         220.34580, 221.32331, 222.30081, 223.27831, 224.25582, 225.23332, 226.21083, 227.18833, 228.16584,
         229.14334, 230.12085, 178.31310]) #THz

c = 299.792458 #  um/ps

ls = c / freqs

for num in range(0, len(power_of_two_hfss)):
    f = interpolate.RectBivariateSpline(y_data_hfss, x_data_hfss, power_of_two_hfss[num])
    xmin = x_data_hfss.min()
    xmax = x_data_hfss.max()
    ymin = y_data_hfss.min()
    ymax = y_data_hfss.max()
    integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
    nominator_hfss[num] = integral ** 2

print("HFSS NOMINATOR CALCULATION DONE!")

for num in range(0, len(power_of_four_hfss)):
    f = interpolate.RectBivariateSpline(y_data_hfss, x_data_hfss, power_of_four_hfss[num])
    xmin = x_data_hfss.min()
    xmax = x_data_hfss.max()
    ymin = y_data_hfss.min()
    ymax = y_data_hfss.max()
    integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
    denominator_hfss[num] = integral

print("HFSS DENOMINATOR CALCULATION DONE!")

Aeff_hfss = nominator_hfss / denominator_hfss

for num in range(0, len(power_of_two)):
    f = interpolate.RectBivariateSpline(y_data, x_data, power_of_two[num])
    xmin = x_data.min()
    xmax = x_data.max()
    ymin = y_data.min()
    ymax = y_data.max()
    integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
    nominator[num] = integral ** 2

print("Finished calculating nominators for TE.")

for num in range(0, len(power_of_four)):
    f = interpolate.RectBivariateSpline(y_data, x_data, power_of_four[num])
    xmin = x_data.min()
    xmax = x_data.max()
    ymin = y_data.min()
    ymax = y_data.max()
    integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
    denominator[num] = integral

print("Finished calculating denominators for TE.")

for num in range(0, len(power_of_two_tm)):
    f = interpolate.RectBivariateSpline(y_data_tm, x_data_tm, power_of_two_tm[num])
    xmin = x_data_tm.min()
    xmax = x_data_tm.max()
    ymin = y_data_tm.min()
    ymax = y_data_tm.max()
    integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
    nominator_tm[num] = integral ** 2

print("Finished calculating nominators for TM.")

for num in range(0, len(power_of_four_tm)):
    f = interpolate.RectBivariateSpline(y_data_tm, x_data_tm, power_of_four_tm[num])
    xmin = x_data_tm.min()
    xmax = x_data_tm.max()
    ymin = y_data_tm.min()
    ymax = y_data_tm.max()
    integral, error = dblquad(lambda x, y: f(x, y), xmin, xmax, lambda x: ymin, lambda x: ymax)
    denominator_tm[num] = integral

print("Finished calculating denominators for TM.")

Aeff = nominator / denominator
Aeff_tm = nominator_tm / denominator_tm

plt.figure(figsize=(10, 6))
plt.scatter(np.arange(1.3, 1.85, 0.01), Aeff, marker='o', label='quasi-TE οnline')
plt.scatter(np.arange(1.55, 1.86, 0.01), Aeff_tm, marker='o', color='red', label='quasi-TM οnline')
plt.scatter(ls, Aeff_hfss, marker='o', color='black', label='quasi-TE HFSS')

plt.xlabel("Wavelength [μm]", fontsize=22) #Common x label
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.ylabel("A$_e$$_f$$_f$ [μm$^2$]", fontsize=22)
plt.grid(True)
plt.legend(fontsize=22)
plt.show()

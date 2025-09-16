import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import math

# Function for configuring the run
def get_config():
    """
    Configuration function defining several variables that are used from the other functions

    Returns:
        dictionary: Variable and its respective value
    """
    return {
        "wavelength_start" : 1.3, #um
        "wavelength_end": 1.9, #um
        "wavelength_no1": 1.31, #um
        "wavelength_no2": 1.55, #um
        "waveguide_width": 0.4, #um
        "waveguide_height": 0.3, #um
        "font": 20, # Fontsize of the titles/axis-names etc in plots
        "min_val_y": -0.6, #um
        "max_val_y": 0.88, #um
        "min_val_x": -1, #um
        "max_val_x": 1, #um
        "c0": 299.792458, #um/ps
        "verbose": True, # Controls the verbocity of the code
        "plot_live": True, # Plot during runtime or save the figures
    }


# Define the n_si equation
def n_si(x):
    """
    Function that takes the wavelength (or an array of wavelengths) and returns the refractive index of silicon.

    Returns:
        The refractive index/indexes of silicon for specific wavelength(s).
    """
    denom1 = 1 - (0.301516485 / x)**2
    denom2 = 1 - (1.13475115 / x)**2
    denom3 = 1 - (1104 / x)**2
    term1 = 1
    term2 = 10.6684293 / denom1
    term3 = 0.0030434748 / denom2
    term4 = 1.54133408 / denom3
    return np.sqrt(term1 + term2 + term3 + term4)

# Define the n_sio2 equation
def n_sio2(x):
    """
    Function that takes the wavelength (or an array of wavelengths) and returns the refractive index of silicon dioxide.

    Returns:
        The refractive index/indexes of silicon dioxide for specific wavelength(s).
    """
    denom1 = 1 - (0.0684043 / x)**2
    denom2 = 1 - (0.1162414 / x)**2
    denom3 = 1 - (9.896161 / x)**2
    term1 = 1
    term2 = 0.6961663 / denom1
    term3 = 0.4079426 / denom2
    term4 = 0.8974794 / denom3

    return np.sqrt(term1 + term2 + term3 + term4)

# Plot the refractive index of silicon for a specific wavelength range
def plot_refractive_si(lower_w, upper_w, step, ax2):
    """
    Function that plots the refractive index of silicon.

    Returns:
        The respective axis containing the plot.
    """
    all_values = np.arange(lower_w, upper_w, step)
    n_si_values = np.zeros(shape=int((upper_w - lower_w) / step) + 1)
    for i, x  in enumerate(all_values):
        n_si_values[i] = n_si(x)
    ax2.plot(all_values * 1000, n_si_values, linestyle='-', color='black')
    ax2.set_xlabel("Wavelength [nm]", fontsize=22)
    ax2.set_ylabel("Refractive Index Si", fontsize=22)
    ax2.tick_params(labelsize=22)
    ax2.grid(True)
    ax2.set_title("Refractive Index of Silicon", fontsize=22)

    return ax2

# Plot the refractive index of silicon dioxide for a specific range
def plot_refractive_sio2(lower_w, upper_w, step, ax1):
    """
    Function that plots the refractive index of silicon dioxide.

    Returns:
        The respective axis containing the plot.
    """
    all_values = np.arange(lower_w, upper_w, step)
    n_sio2_values = np.zeros(shape=int((upper_w - lower_w) / step) + 1)
    for i, x in enumerate(all_values):
        n_sio2_values[i] = n_sio2(x)
    ax1.plot(all_values * 1000, n_sio2_values, linestyle='-', color='black')
    ax1.set_xlabel("Wavelength [nm]", fontsize=22)
    ax1.set_ylabel(r"Refractive Index SiO$_2$", fontsize=22)
    ax1.tick_params(labelsize=22)
    ax1.grid(True)
    ax1.set_title("Refractive Index of Silicon Dioxide", fontsize=22)

    return ax1


def plot_profile_online(filename, variable, mode, config, ax, fig):
    """
    Function that plots the profile from the online simulator output.
    """
    data = np.loadtxt(filename)
    y = data[:, 0]
    x = data[:, 1]
    intensity = data[:, 2]

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    x_grid_size = len(x_unique)
    y_grid_size = len(y_unique)

    intensity_grid = intensity.reshape(y_grid_size, x_grid_size)

    mesh = ax.pcolormesh(x_unique, y_unique, intensity_grid, shading='gouraud', cmap="magma")

    cbar = fig.colorbar(mesh, ax=ax, extend='both')
    cbar.set_label(variable, fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=3)  # Bold

    y_min_norm = (0 - min(y_unique)) / (max(y_unique) - min(y_unique))
    y_max_norm = (config["waveguide_height"] - min(y_unique)) / (max(y_unique) - min(y_unique))
    x_min_norm = (-config["waveguide_width"] / 2 - min(x_unique)) / (max(x_unique) - min(x_unique))
    x_max_norm = (config["waveguide_width"] / 2 - min(x_unique)) / (max(x_unique) - min(x_unique))

    ax.axvline(x=config["waveguide_width"] / 2, color='black', linestyle='-', linewidth=3, ymin=y_min_norm, ymax=y_max_norm)
    ax.axvline(x=-config["waveguide_width"] / 2, color='black', linestyle='-', linewidth=3, ymin=y_min_norm, ymax=y_max_norm)
    ax.axhline(y=config["waveguide_height"], color='black', linestyle='-', linewidth=3, xmin=x_min_norm, xmax=x_max_norm)

    ax.set_xlabel('x [μm]')
    ax.set_ylabel('y [μm]')
    ax.set_title('Online Tool')
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    return ax

def plot_profile_tidy3d(ax_e, min_val_y, min_val_x, max_val_y, max_val_x, variable, mode, config):
    """
    Function that plots the profile from the tidy3d simulator output.
    """
    ax_e.axhline(y=0, color='black', linestyle='-', linewidth=3)
    ax_e.set_ylim(min_val_y, max_val_y)
    ax_e.set_xlim(min_val_x, max_val_x)
    y_min_norm = (0 - min_val_y) / (max_val_y - min_val_y)
    y_max_norm = (config["waveguide_height"] - min_val_y) / (max_val_y - min_val_y)
    x_min_norm = (-config["waveguide_width"] / 2 - min_val_x) / (max_val_x - min_val_x)
    x_max_norm = (config["waveguide_width"] / 2 - min_val_x) / (max_val_x - min_val_x)
    ax_e.axvline(x=config["waveguide_width"] / 2, color='black', linestyle='-', linewidth=3, ymin=y_min_norm, ymax=y_max_norm)
    ax_e.axvline(x=-config["waveguide_width"] / 2, color='black', linestyle='-', linewidth=3, ymin=y_min_norm, ymax=y_max_norm)
    ax_e.axhline(y=config["waveguide_height"], color='black', linestyle='-', linewidth=3, xmin=x_min_norm, xmax=x_max_norm)
    ax_e.set_xlabel('x [μm]')
    ax_e.set_ylabel('y [μm]')
    ax_e.set_title('Tidy3d', fontsize=20)

    return ax_e


def plot_profile_hfss(csv_file, min_val_y, min_val_x, max_val_y, max_val_x, variable, mode, config, ax ,fig):

    df = pd.read_csv(csv_file, header=None, names=['x', 'y', variable])

    min_val_yn = min_val_y + 2
    max_val_yn = max_val_y + 2
    min_val_xn = min_val_x + 1
    max_val_xn = max_val_x + 1

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df[variable] = pd.to_numeric(df[variable], errors='coerce')

    # Filter the DataFrame to include only points within the bounding box
    df_filtered = df[
        (df['x'] >= min_val_xn) & (df['x'] <= max_val_xn) &
        (df['y'] >= min_val_yn) & (df['y'] <= max_val_yn)
    ]

    # Create the profile plot

    dfx = np.unique(df_filtered['x'].values)
    dfy = np.unique(df_filtered['y'].values)
    dfi = df[variable].values * 1e-6
    dfi = dfi[1:].reshape(len(dfy), len(dfx))

    dfx = np.linspace(min_val_x, max_val_x, len(dfx))
    dfy = np.linspace(min_val_y, max_val_y, len(dfy))

    # Scatter plot of intensity values
    mesh = ax.pcolormesh(dfx, dfy, dfi, shading='gouraud', cmap="magma")

    cbar = fig.colorbar(mesh, ax=ax, extend='both')
    cbar.set_label(variable, fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=3)  # Bold

    y_min_norm = (0 - min(dfy)) / (max(dfy) - min(dfy))
    y_max_norm = (config["waveguide_height"] - min(dfy)) / (max(dfy) - min(dfy))
    x_min_norm = (-config["waveguide_width"] / 2 - min(dfx)) / (max(dfx) - min(dfx))
    x_max_norm = (config["waveguide_width"] / 2 - min(dfx)) / (max(dfx) - min(dfx))

    ax.axvline(x=config["waveguide_width"] / 2, color='black', linestyle='-', linewidth=3, ymin=y_min_norm,
               ymax=y_max_norm)
    ax.axvline(x=-config["waveguide_width"] / 2, color='black', linestyle='-', linewidth=3, ymin=y_min_norm,
               ymax=y_max_norm)
    ax.axhline(y=config["waveguide_height"], color='black', linestyle='-', linewidth=3, xmin=x_min_norm,
               xmax=x_max_norm)

    ax.set_xlabel('x [μm]')
    ax.set_ylabel('y [μm]')
    ax.set_title('HFSS')
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

def plot_all_profiles(file_hfss, file_online, mode, field, strip, config):

    if mode == "Quasi-TE":
        mode_index = 0
    else:
        mode_index = 1

    if file_online == None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 15))
        ax_1p31_tidy_te_e = strip.plot_field(field, mode_index=mode_index, val="abs", ax=axes[0])
        plot_profile_tidy3d(ax_1p31_tidy_te_e, config["min_val_y"], config["min_val_x"], config["max_val_y"],
                            config["max_val_x"],
                            "|" + field + "|", mode, config)
        asp = np.diff(ax_1p31_tidy_te_e.get_xlim())[0] / np.diff(ax_1p31_tidy_te_e.get_ylim())[0]
        plot_profile_hfss(file_hfss, config["min_val_y"], config["min_val_x"], config["max_val_y"],
                          config["max_val_x"], "|" + field + "|", mode, config, axes[1], fig)
        axes[0].set_aspect(asp)
        axes[1].set_aspect(asp)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(6, 15))
        ax_1p31_tidy_te_e = strip.plot_field(field, mode_index=mode_index, val="abs", ax=axes[0])
        plot_profile_tidy3d(ax_1p31_tidy_te_e, config["min_val_y"], config["min_val_x"], config["max_val_y"],
                            config["max_val_x"],
                            "|" + field + "|", mode, config)
        asp = np.diff(ax_1p31_tidy_te_e.get_xlim())[0] / np.diff(ax_1p31_tidy_te_e.get_ylim())[0]
        plot_profile_online(file_online, "|" + field + "|", mode, config, axes[1], fig)
        plot_profile_hfss(file_hfss, config["min_val_y"], config["min_val_x"], config["max_val_y"],
                          config["max_val_x"], "|" + field + "|", mode, config, axes[2], fig)
        axes[0].set_aspect(asp)
        axes[1].set_aspect(asp)
        axes[2].set_aspect(asp)

    fig.subplots_adjust(wspace=0.75)

    if config["plot_live"]:
        plt.show()
    else:
        plt.savefig("./profiles/" + mode + "_" + field + ".png", format="png")


def plot_ex1(strip_1p31, strip_1p55, config):

    plot_all_profiles("./HFSS_ex1/TE/TE_Emag_1p31.csv",
                      "./online_ex1/te0-0aE_1p31.xyz",
                      "Quasi-TE",
                      "E",
                      strip_1p31,
                      config)

    plot_all_profiles("./HFSS_ex1/TE/TE_Exmag_1p31.csv",
                      "./online_ex1/te0-0Eymod_1p31.xyz",
                      "Quasi-TE",
                      "Ex",
                      strip_1p31,
                      config)

    plot_all_profiles("./HFSS_ex1/TE/TE_Eymag_1p31.csv",
                      "./online_ex1/te0-0Exmod_1p31.xyz",
                      "Quasi-TE",
                      "Ey",
                      strip_1p31,
                      config)

    plot_all_profiles("./HFSS_ex1/TE/TE_Ezmag_1p31.csv",
                      "./online_ex1/te0-0Ezmod_1p31.xyz",
                      "Quasi-TE",
                      "Ez",
                      strip_1p31,
                      config)

    plot_all_profiles("./HFSS_ex1/TM/TM_Emag_1p31.csv",
                      None,
                      "Quasi-TM",
                      "E",
                      strip_1p31,
                      config)

    plot_all_profiles("./HFSS_ex1/TM/TM_Exmag_1p31.csv",
                      None,
                      "Quasi-TM",
                      "Ex",
                      strip_1p31,
                      config)

    plot_all_profiles("./HFSS_ex1/TM/TM_Eymag_1p31.csv",
                      None,
                      "Quasi-TM",
                      "Ey",
                      strip_1p31,
                      config)

    plot_all_profiles("./HFSS_ex1/TM/TM_Ezmag_1p31.csv",
                      None,
                      "Quasi-TM",
                      "Ez",
                      strip_1p31,
                      config)

    ###

    plot_all_profiles("./HFSS_ex1/TE/TE_Emag_1p55.csv",
                      "./online_ex1/te0-0aE_1p55.xyz",
                      "Quasi-TE",
                      "E",
                      strip_1p55,
                      config)

    plot_all_profiles("./HFSS_ex1/TE/TE_Exmag_1p55.csv",
                      "./online_ex1/te0-0Eymod_1p55.xyz",
                      "Quasi-TE",
                      "Ex",
                      strip_1p55,
                      config)

    plot_all_profiles("./HFSS_ex1/TE/TE_Eymag_1p55.csv",
                      "./online_ex1/te0-0Exmod_1p55.xyz",
                      "Quasi-TE",
                      "Ey",
                      strip_1p55,
                      config)

    plot_all_profiles("./HFSS_ex1/TE/TE_Ezmag_1p55.csv",
                      "./online_ex1/te0-0Ezmod_1p55.xyz",
                      "Quasi-TE",
                      "Ez",
                      strip_1p55,
                      config)

    plot_all_profiles("./HFSS_ex1/TM/TM_Emag_1p55.csv",
                      "./online_ex1/tm0-0aE_1p55.xyz",
                      "Quasi-TM",
                      "E",
                      strip_1p55,
                      config)

    plot_all_profiles("./HFSS_ex1/TM/TM_Exmag_1p55.csv",
                      "./online_ex1/tm0-0Eymod_1p55.xyz",
                      "Quasi-TM",
                      "Ex",
                      strip_1p55,
                      config)

    plot_all_profiles("./HFSS_ex1/TM/TM_Eymag_1p55.csv",
                      "./online_ex1/tm0-0Exmod_1p55.xyz",
                      "Quasi-TM",
                      "Ey",
                      strip_1p55,
                      config)

    plot_all_profiles("./HFSS_ex1/TM/TM_Ezmag_1p55.csv",
                      "./online_ex1/tm0-0Ezmod_1p55.xyz",
                      "Quasi-TM",
                      "Ez",
                      strip_1p55,
                      config)

def group_index_calc(n_eff, lambdas):
    dneffdlambdas = np.gradient(n_eff, lambdas)
    ng = n_eff - lambdas * dneffdlambdas
    return ng


def process_data_from_directory(directory, starter, ender):

    z_arrays = []
    x_coords = None
    y_coords = None
    first_file_processed = False

    for filename in os.listdir(directory):
        if filename.startswith(starter) and filename.endswith(ender):  # Adjust extension if needed
            filepath = os.path.join(directory, filename)
            try:
                data = np.genfromtxt(filepath, delimiter='  ')  # Two spaces as delimiter

                if data.ndim != 2 or data.shape[1] != 3:
                    print(f"Warning: File '{filename}' does not have the expected format. Skipping.")
                    continue

                z_arrays.append(data[:, 2].reshape(75, 100))  # Extract third column (values)

                if not first_file_processed:
                    x_coords = data[:, 1]
                    y_coords = data[:, 0]
                    x_coords_uniq = np.zeros(shape=(100))
                    y_coords_uniq = np.zeros(shape=(75))
                    # Get only the unique values
                    i = 0
                    for x in x_coords:
                        if x not in x_coords_uniq:
                            x_coords_uniq[i] = x
                            i += 1
                    i = 0
                    for y in y_coords:
                        if y not in y_coords_uniq:
                            y_coords_uniq[i] = y
                            i += 1

                    first_file_processed = True
            except (FileNotFoundError, ValueError) as e:
                print(f"Error processing file '{filename}': {e}")
                return None, None, None  # Return None if any error occurs


    if not z_arrays:
        print("No suitable files found in the directory.")
        return None, None, None

    return z_arrays, x_coords_uniq, y_coords_uniq

def read_hfss_csv(filename, freq_num):
    field_values = []
    all_fields = []
    x_values = []
    y_values = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for row in reader:
            value = float(row[0])
            x_values.append(value)

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for row in reader:
            value = float(row[1])
            y_values.append(value)

    for i in range(2, freq_num):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = next(reader)

            for row in reader:
                value = float(row[i])
                field_values.append(value)

        all_fields.append(np.array(field_values).reshape(11, 20))
        field_values = []

    return all_fields, np.unique(np.array(x_values)), np.unique(np.array(y_values))


def plot_ex2(strip_neff, config):
    # Effective refractive index calculated by the online solver for wavelengths 1.3-1.85 (0.01 step)
    n_on_te = [2.697417304, 2.682655076, 2.667771683, 2.652760797,
               2.637616281, 2.622332110, 2.606902296, 2.591320843,
               2.575581702, 2.559678729, 2.543605653, 2.527356048,
               2.510923297, 2.494300570, 2.477480790, 2.460456606,
               2.443220361, 2.425764059, 2.408079336, 2.390157415,
               2.371989073, 2.353564593, 2.334873719, 2.315905600,
               2.296648735, 2.277090904, 2.257219099, 2.237019437,
               2.216477071, 2.195576082, 2.174299364, 2.152628482,
               2.130543517, 2.108022887, 2.085043135, 2.061578688,
               2.037601564, 2.013081044, 1.987983265, 1.962270747,
               1.935901820, 1.908829923, 1.881002764, 1.852361263,
               1.822838254, 1.792356833, 1.760828264, 1.728149258,
               1.694198404, 1.658831399, 1.621874531, 1.583115598,
               1.542290883, 1.499065900, 1.453005856]

    # Effective index calculation for tidy3d
    n_eff = strip_neff.mode_solver.data.n_eff
    n_eff_mode0 = n_eff[:, 0]
    lambdas_tidy3d = config["c0"] / (np.asarray(n_eff_mode0['f']) * 1e-15)

    # Calculation of effective index from Gamma values (phase constant) from HFSS
    df = pd.read_csv("Gamma.csv")
    y_column = df.columns[0]
    x1_column = df.columns[1]
    hfss_x = (config["c0"] / df[y_column]) * 1e3
    k = 2 * math.pi / (config["c0"] / df[y_column])
    hfss_n_eff_0 = [(df[x1_column][x] * 1e-6) / k[x] for x in range(0, len(df[x1_column]))]

    lambdas = np.linspace(1300, 1840, 55)

    plt.plot(lambdas_tidy3d, n_eff_mode0, linestyle='-', color='green', label='quasi-TE Tidy3d')
    plt.scatter(lambdas, n_on_te, marker='o', color='blue', label='quasi-TE online')
    plt.plot(hfss_x, hfss_n_eff_0, linestyle='--', color='black', label='quasi-TE HFSS')

    plt.xlabel("Wavelength [nm]")
    plt.ylabel(r'$n_{eff}$')
    plt.title("Mode 0")
    plt.legend()
    plt.grid(True)
    if config["plot_live"]:
        plt.show()
    else:
        plt.savefig("./neff_mode0.png", format="png")

    n_eff_mode1 = n_eff[:, 1]
    lambdas_tidy3d = config["c0"] / (np.asarray(n_eff_mode1['f']) * 1e-15)
    x2_column = df.columns[2]
    k = 2 * math.pi / (config["c0"] / df[y_column])
    hfss_n_eff_1 = [(df[x2_column][x] * 1e-6) / k[x] for x in range(0, len(df[x2_column]))]
    plt.plot(lambdas_tidy3d, n_eff_mode1, linestyle='-', color='green', label='quasi-TM Tidy3d')
    plt.plot(hfss_x, hfss_n_eff_1, linestyle='--', color='black', label='quasi-TM HFSS')

    plt.xlabel("Wavelength [nm]")
    plt.ylabel(r'$n_{eff}$')
    plt.title("Mode 1")
    plt.legend()
    plt.grid(True)
    if config["plot_live"]:
        plt.show()
    else:
        plt.savefig("./neff_mode1.png", format="png")

    return n_on_te, hfss_n_eff_0, hfss_n_eff_1, lambdas, hfss_x

def plot_ex3(strip_neff, config, n_on_te, lambdas, hfss_n_eff_0, hfss_x, hfss_n_eff_1):

    ng_tidy3d = strip_neff.mode_solver.data.n_group

    ng_mode0 = ng_tidy3d[:, 0]
    lambdas_tidy3d = config["c0"] / (np.asarray(ng_mode0['f']) * 1e-15)
    plt.plot(lambdas_tidy3d, ng_mode0, linestyle='-', color='green', label='quasi-TE Tidy3d')

    ng_online = group_index_calc(n_on_te, lambdas)

    ng_hfss_mode0 = group_index_calc(hfss_n_eff_0[::-1], hfss_x[::-1])

    plt.scatter(lambdas, ng_online, marker='o', color='blue', label='quasi-TE online')
    plt.plot(hfss_x, ng_hfss_mode0[::-1], linestyle='--', color='black', label='quasi-TE HFSS')

    plt.xlabel("Wavelength [nm]")
    plt.ylabel(r'$n_{g}$')
    plt.title("Mode 0")
    plt.legend()
    plt.grid(True)
    if config["plot_live"]:
        plt.show()
    else:
        plt.savefig("./ng_mode0.png", format="png")

    ng_mode1 = ng_tidy3d[:, 1]
    lambdas_tidy3d = config["c0"] / (np.asarray(ng_mode1['f']) * 1e-15)
    plt.plot(lambdas_tidy3d, ng_mode1, linestyle='-', color='green', label='quasi-TM Tidy3d')

    ng_hfss_mode1 = group_index_calc(hfss_n_eff_1[::-1], hfss_x[::-1])
    plt.plot(hfss_x, ng_hfss_mode1[::-1], linestyle='--', color='black', label='quasi-TM HFSS')

    plt.xlabel("Wavelength [nm]")
    plt.ylabel(r'$n_{g}$')
    plt.title("Mode 1")
    plt.legend()
    plt.grid(True)
    if config["plot_live"]:
        plt.show()
    else:
        plt.savefig("./ng_mode1.png", format="png")



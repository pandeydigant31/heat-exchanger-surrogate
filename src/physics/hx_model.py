"""Shell-and-tube heat exchanger model using the ε-NTU method.

Computes temperature profiles T_hot(x), T_cold(x) along the HX length,
outlet temperatures, heat transfer rate, and effectiveness.

Uses CoolProp for temperature-dependent fluid properties (water, air, glycol).
"""

import numpy as np
import CoolProp.CoolProp as CP


# ── Fluid properties via CoolProp ──────────────────────────────

def fluid_props(T_K, P_Pa, fluid="Water"):
    """Get fluid properties at given temperature and pressure."""
    return {
        "cp": CP.PropsSI("C", "T", T_K, "P", P_Pa, fluid),          # J/kg/K
        "rho": CP.PropsSI("D", "T", T_K, "P", P_Pa, fluid),         # kg/m³
        "mu": CP.PropsSI("V", "T", T_K, "P", P_Pa, fluid),          # Pa·s
        "k": CP.PropsSI("L", "T", T_K, "P", P_Pa, fluid),           # W/m/K
        "Pr": CP.PropsSI("Prandtl", "T", T_K, "P", P_Pa, fluid),    # -
    }


# ── Heat transfer correlations ─────────────────────────────────

def dittus_boelter(Re, Pr, heating=True):
    """Dittus-Boelter correlation for internal forced convection.

    Nu = 0.023 * Re^0.8 * Pr^n  (n=0.4 heating, n=0.3 cooling)
    Valid for Re > 10,000 and 0.6 < Pr < 160.
    """
    n = 0.4 if heating else 0.3
    return 0.023 * Re**0.8 * Pr**n


def compute_UA(m_dot_hot, m_dot_cold, T_hot_avg, T_cold_avg,
               D_tube, D_shell, L, n_tubes, fouling=0.0001):
    """Compute overall heat transfer coefficient × area (UA).

    Simple shell-and-tube model:
        - Tube side: hot fluid (Dittus-Boelter)
        - Shell side: cold fluid (simplified Kern method)
    """
    P = 101325.0  # Pa (atmospheric)

    # Hot side (tube) properties at average temp
    props_h = fluid_props(T_hot_avg + 273.15, P)
    A_tube = np.pi * (D_tube / 2) ** 2
    v_tube = m_dot_hot / (props_h["rho"] * A_tube * n_tubes)
    Re_tube = props_h["rho"] * v_tube * D_tube / props_h["mu"]
    Re_tube = max(Re_tube, 100)  # floor for stability
    Nu_tube = max(dittus_boelter(Re_tube, props_h["Pr"], heating=False), 3.66)
    h_tube = Nu_tube * props_h["k"] / D_tube

    # Cold side (shell) — simplified
    props_c = fluid_props(T_cold_avg + 273.15, P)
    # Shell-side hydraulic diameter (simplified)
    D_shell_hyd = D_shell - n_tubes * D_tube
    D_shell_hyd = max(D_shell_hyd, 0.01)
    A_shell = np.pi * (D_shell / 2) ** 2 - n_tubes * np.pi * (D_tube / 2) ** 2
    A_shell = max(A_shell, 0.001)
    v_shell = m_dot_cold / (props_c["rho"] * A_shell)
    Re_shell = props_c["rho"] * v_shell * D_shell_hyd / props_c["mu"]
    Re_shell = max(Re_shell, 100)
    Nu_shell = max(dittus_boelter(Re_shell, props_c["Pr"], heating=True), 3.66)
    h_shell = Nu_shell * props_c["k"] / D_shell_hyd

    # Overall U (tube-based area)
    A_total = np.pi * D_tube * L * n_tubes  # total tube surface area
    R_tube = 1 / (h_tube * A_total)
    R_shell = 1 / (h_shell * A_total)
    R_fouling = fouling / A_total
    R_wall = 0.001 / (50 * A_total)  # thin steel wall, k=50 W/m/K

    UA = 1 / (R_tube + R_shell + R_fouling + R_wall)
    return UA, A_total


# ── ε-NTU method ───────────────────────────────────────────────

def effectiveness_counterflow(NTU, C_r):
    """Effectiveness for counterflow heat exchanger.

    ε = (1 - exp(-NTU(1-Cr))) / (1 - Cr*exp(-NTU(1-Cr)))
    Special case: Cr=1 → ε = NTU/(1+NTU)
    """
    if C_r < 0.999:
        exp_term = np.exp(-NTU * (1 - C_r))
        return (1 - exp_term) / (1 - C_r * exp_term)
    else:
        return NTU / (1 + NTU)


def solve_hx(T_hot_in, T_cold_in, m_dot_hot, m_dot_cold,
             D_tube=0.02, D_shell=0.15, L=2.0, n_tubes=20,
             fouling=0.0001, n_points=50):
    """Solve heat exchanger using ε-NTU method.

    Returns temperature profiles along the HX length and summary metrics.

    Args:
        T_hot_in: hot fluid inlet temperature (°C)
        T_cold_in: cold fluid inlet temperature (°C)
        m_dot_hot: hot side mass flow rate (kg/s)
        m_dot_cold: cold side mass flow rate (kg/s)
        D_tube: tube inner diameter (m)
        D_shell: shell inner diameter (m)
        L: tube length (m)
        n_tubes: number of tubes
        fouling: fouling factor (m²K/W)
        n_points: spatial discretization

    Returns:
        dict with profiles and metrics
    """
    P = 101325.0

    # Average temperatures (initial estimate)
    T_hot_avg = (T_hot_in + T_cold_in) / 2 + 10
    T_cold_avg = (T_hot_in + T_cold_in) / 2 - 10

    # Capacity rates
    cp_hot = CP.PropsSI("C", "T", T_hot_avg + 273.15, "P", P, "Water")
    cp_cold = CP.PropsSI("C", "T", T_cold_avg + 273.15, "P", P, "Water")

    C_hot = m_dot_hot * cp_hot    # W/K
    C_cold = m_dot_cold * cp_cold  # W/K

    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    C_r = C_min / C_max

    # Overall UA
    UA, A_total = compute_UA(m_dot_hot, m_dot_cold, T_hot_avg, T_cold_avg,
                              D_tube, D_shell, L, n_tubes, fouling)

    # NTU and effectiveness
    NTU = UA / C_min
    eps = effectiveness_counterflow(NTU, C_r)

    # Heat transfer
    Q_max = C_min * (T_hot_in - T_cold_in)
    Q = eps * Q_max

    # Outlet temperatures
    T_hot_out = T_hot_in - Q / C_hot
    T_cold_out = T_cold_in + Q / C_cold

    # Temperature profiles along length (counterflow approximation)
    x = np.linspace(0, L, n_points)
    # Linear interpolation for profiles (simplified)
    T_hot_profile = T_hot_in - (T_hot_in - T_hot_out) * (x / L)
    T_cold_profile = T_cold_out - (T_cold_out - T_cold_in) * (x / L)

    return {
        "x": x,
        "T_hot": T_hot_profile,
        "T_cold": T_cold_profile,
        "T_hot_out": T_hot_out,
        "T_cold_out": T_cold_out,
        "Q": Q,
        "effectiveness": eps,
        "NTU": NTU,
        "UA": UA,
        "LMTD": Q / UA if UA > 0 else 0,
        # Input params (for dataset)
        "T_hot_in": T_hot_in,
        "T_cold_in": T_cold_in,
        "m_dot_hot": m_dot_hot,
        "m_dot_cold": m_dot_cold,
        "fouling": fouling,
        "L": L,
        "n_tubes": n_tubes,
    }


def generate_dataset(n_samples=5000, n_points=50, seed=42):
    """Generate training dataset by varying HX operating conditions.

    Varies: inlet temps, flow rates, geometry, fouling.
    """
    np.random.seed(seed)

    inputs = []
    T_hot_profiles = []
    T_cold_profiles = []
    outputs = []

    for i in range(n_samples):
        T_hot_in = np.random.uniform(60, 95)     # °C
        T_cold_in = np.random.uniform(10, 35)     # °C
        m_dot_hot = np.random.uniform(0.5, 5.0)   # kg/s
        m_dot_cold = np.random.uniform(0.5, 5.0)  # kg/s
        L = np.random.uniform(1.0, 4.0)           # m
        n_tubes = np.random.randint(10, 40)
        fouling = np.random.uniform(0.0, 0.0005)  # m²K/W

        try:
            result = solve_hx(T_hot_in, T_cold_in, m_dot_hot, m_dot_cold,
                              L=L, n_tubes=n_tubes, fouling=fouling, n_points=n_points)
        except Exception:
            continue

        # Validate physics
        if (result["T_hot_out"] < T_cold_in or result["T_cold_out"] > T_hot_in
                or result["Q"] < 0 or result["effectiveness"] > 1.0
                or result["effectiveness"] < 0):
            continue

        inputs.append([T_hot_in, T_cold_in, m_dot_hot, m_dot_cold, L, n_tubes, fouling])
        T_hot_profiles.append(result["T_hot"])
        T_cold_profiles.append(result["T_cold"])
        outputs.append([result["T_hot_out"], result["T_cold_out"],
                        result["Q"], result["effectiveness"]])

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{n_samples} ({len(inputs)} valid)", flush=True)

    return {
        "inputs": np.array(inputs, dtype=np.float32),
        "T_hot_profiles": np.array(T_hot_profiles, dtype=np.float32),
        "T_cold_profiles": np.array(T_cold_profiles, dtype=np.float32),
        "outputs": np.array(outputs, dtype=np.float32),
        "n_points": n_points,
        "input_names": ["T_hot_in", "T_cold_in", "m_dot_hot", "m_dot_cold",
                        "L", "n_tubes", "fouling"],
        "output_names": ["T_hot_out", "T_cold_out", "Q", "effectiveness"],
    }

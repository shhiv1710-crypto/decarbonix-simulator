
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

# ============================================================
# CARBONOS CONTROL TOWER
# DECARBONIX 1.0 - PROBLEM 02
# Data + constraints are taken from the supplied final-round
# problem statement. The dispatch is solved as a MILP.
# ============================================================

st.set_page_config(page_title="CarbonOS Control Tower", page_icon="🌱", layout="wide")

st.title("🌱 CarbonOS — Control Tower")
st.subheader("From Target → Feasible Shift Plan")

# ============================================================
# 1. COMPLETE 24-HOUR INPUT DATA FROM TABLE 5
# ============================================================

hours = np.arange(24)
hour_labels = [f"{h:02d}:00-{(h+1)%24:02d}:00" for h in hours]

steam_demand = np.array([
    92, 88, 86, 84, 86, 92, 100, 110, 122, 132, 140, 146,
    150, 148, 145, 142, 138, 134, 128, 120, 112, 106, 100, 96
], dtype=float)

waste_heat_avail = np.array([
    8, 8, 8, 8, 8, 10, 12, 15, 18, 20, 20, 20,
    20, 20, 20, 18, 18, 16, 14, 12, 10, 10, 8, 8
], dtype=float)

fixed_electricity = np.array([
    8.5, 8.2, 8.0, 8.1, 8.4, 9.0, 9.8, 10.5,
    11.2, 11.8, 12.4, 12.8, 13.0, 13.2, 13.0,
    12.7, 12.5, 12.2, 11.8, 11.5, 10.8, 10.0, 9.4, 8.9
], dtype=float)

baseline_flexible = np.array([
    0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0, 0, 0, 0, 0
], dtype=float)

solar_availability = np.array([
    0, 0, 0, 0, 0, 0, 0.5, 2, 4, 6, 8, 9.5,
    10, 9.5, 8.5, 7, 5, 3, 1, 0, 0, 0, 0, 0
], dtype=float)

grid_factor = np.array([
    0.78, 0.76, 0.74, 0.72, 0.70, 0.68,
    0.66, 0.64, 0.60, 0.56, 0.52, 0.48,
    0.46, 0.45, 0.47, 0.50, 0.55, 0.62,
    0.70, 0.78, 0.82, 0.84, 0.82, 0.80
], dtype=float)

tariff = np.array([
    5200, 5200, 5200, 5200, 5200, 5200,
    6800, 6800, 6800, 6800,
    5600, 5600, 5600, 5600, 5600, 5600,
    7200, 7200, 7200,
    8200, 8200, 8200, 8200,
    6000
], dtype=float)

# ============================================================
# 2. OFFICIAL BASELINE + HARD LIMITS
# ============================================================

BASELINE_CO2 = 986.992
BASELINE_COST = 5_526_720

COAL_FUEL_COEFF = 0.18       # t coal / t steam
BIO_FUEL_COEFF = 0.27        # t biomass / t steam
GAS_FUEL_COEFF = 80.0        # Sm3 gas / t steam

COAL_EF = 2.35               # tCO2e / t coal
BIO_EF = 0.10                # tCO2e / t biomass
GAS_EF = 0.00200             # tCO2e / Sm3 gas

COAL_PRICE = 8500            # INR / t coal
BIO_PRICE = 5500             # INR / t biomass
GAS_PRICE = 32               # INR / Sm3 gas

COAL_DAILY_LIMIT = 350       # t/day
BIO_DAILY_LIMIT = 220        # t/day
GAS_DAILY_LIMIT = 20_000     # Sm3/day
GRID_LIMIT = 22              # MW

COAL_MIN = 25
COAL_MAX = 90
COAL_RAMP = 10
COAL_MIN_ON = 4

BIO_MIN = 20
BIO_MAX = 65
BIO_RAMP = 8
BIO_MIN_ON = 3

GAS_MAX = 45
GAS_RAMP = 45

FLEX_DAILY_ENERGY = 18.0     # MWh/day
FLEX_MAX = 2.5               # MW
FLEX_START_HOUR = 6
FLEX_END_HOUR = 23           # inclusive
CONTINUITY_START = 8
CONTINUITY_END = 18          # inclusive
CONTINUITY_MIN = 0.5         # MW

# ============================================================
# 3. BASELINE RECONSTRUCTION
# ============================================================

def reconstruct_official_baseline():
    # Exactly the deterministic rule in the problem statement.
    wh = np.minimum(10.0, waste_heat_avail)
    bio = np.full(24, 20.0)
    remaining = steam_demand - bio - wh
    coal = np.minimum(np.maximum(remaining, 0), 90.0)
    gas = np.maximum(remaining - coal, 0)

    solar_used = 0.80 * solar_availability
    grid = fixed_electricity + baseline_flexible - solar_used

    coal_fuel = coal * COAL_FUEL_COEFF
    bio_fuel = bio * BIO_FUEL_COEFF
    gas_fuel = gas * GAS_FUEL_COEFF

    steam_emissions = (
        coal_fuel * COAL_EF
        + bio_fuel * BIO_EF
        + gas_fuel * GAS_EF
    )
    grid_emissions = grid * grid_factor
    emissions = steam_emissions + grid_emissions

    cost = (
        coal_fuel * COAL_PRICE
        + bio_fuel * BIO_PRICE
        + gas_fuel * GAS_PRICE
        + grid * tariff
    )

    return {
        "coal": coal, "bio": bio, "gas": gas, "wh": wh,
        "solar": solar_used, "grid": grid, "flex": baseline_flexible,
        "emissions": emissions, "cost": cost
    }

# ============================================================
# 4. MILP DISPATCH SOLVER
# ============================================================

VAR_NAMES = [
    "coal", "bio", "gas", "wh", "solar", "grid", "flex",
    "on_coal", "on_bio", "start_coal", "start_bio"
]
IDX = {name: np.arange(24) + k * 24 for k, name in enumerate(VAR_NAMES)}
NVAR = 11 * 24


def solve_dispatch(resilience=False, carbon_cap=None, cost_cap=None):
    """
    Find an hourly feasible dispatch.

    Objective:
        minimise challenge emissions, with a tiny cost tie-breaker.

    Hard constraints:
      - steam balance: demand <= supply <= 103% demand
      - electricity balance: solar + grid = fixed + flexible
      - coal/biomass min/max and ramp
      - coal/biomass minimum-on time
      - gas max/ramp
      - solar availability
      - grid <= 22 MW
      - flexible load exactly 18 MWh/day
      - 0.5 MW continuity during 08:00-18:00
      - daily fuel limits
      - resilience biomass and solar disturbance
    """

    lower = np.zeros(NVAR)
    upper = np.full(NVAR, np.inf)
    integer = np.zeros(NVAR)

    biomass_max = np.full(24, float(BIO_MAX))
    solar_avail = solar_availability.copy()

    if resilience:
        biomass_max[14:18] = 20.0
        solar_avail[12:16] *= 0.60

    upper[IDX["coal"]] = COAL_MAX
    upper[IDX["bio"]] = biomass_max
    upper[IDX["gas"]] = GAS_MAX
    upper[IDX["wh"]] = waste_heat_avail
    upper[IDX["solar"]] = solar_avail
    upper[IDX["grid"]] = GRID_LIMIT
    upper[IDX["flex"]] = FLEX_MAX

    # Flexible production is permitted only from 06:00 through 23:00.
    for h in range(24):
        if not (FLEX_START_HOUR <= h <= FLEX_END_HOUR):
            upper[IDX["flex"][h]] = 0.0

    # Binary variables.
    for name in ["on_coal", "on_bio", "start_coal", "start_bio"]:
        upper[IDX[name]] = 1.0
        integer[IDX[name]] = 1

    rows, lows, highs = [], [], []

    def add_constraint(coeffs, low=-np.inf, high=np.inf):
        rows.append(coeffs)
        lows.append(low)
        highs.append(high)

    for h in range(24):

        # ----- Steam balance -----
        add_constraint({
            IDX["coal"][h]: 1,
            IDX["bio"][h]: 1,
            IDX["gas"][h]: 1,
            IDX["wh"][h]: 1
        }, steam_demand[h], 1.03 * steam_demand[h])

        # ----- Electricity balance -----
        # solar + grid = fixed demand + flexible load
        add_constraint({
            IDX["solar"][h]: 1,
            IDX["grid"][h]: 1,
            IDX["flex"][h]: -1
        }, fixed_electricity[h], fixed_electricity[h])

        # ----- Coal and biomass commitment -----
        add_constraint({
            IDX["coal"][h]: 1,
            IDX["on_coal"][h]: -COAL_MAX
        }, -np.inf, 0)
        add_constraint({
            IDX["coal"][h]: 1,
            IDX["on_coal"][h]: -COAL_MIN
        }, 0, np.inf)

        add_constraint({
            IDX["bio"][h]: 1,
            IDX["on_bio"][h]: -BIO_MAX
        }, -np.inf, 0)
        add_constraint({
            IDX["bio"][h]: 1,
            IDX["on_bio"][h]: -BIO_MIN
        }, 0, np.inf)

        # ----- Ramp constraints -----
        for name, ramp, initial_output in [
            ("coal", COAL_RAMP, 60.0),
            ("bio", BIO_RAMP, 20.0),
            ("gas", GAS_RAMP, 0.0)
        ]:
            if h == 0:
                add_constraint({IDX[name][h]: 1},
                               -np.inf, initial_output + ramp)
                add_constraint({IDX[name][h]: -1},
                               -np.inf, ramp - initial_output)
            else:
                add_constraint({
                    IDX[name][h]: 1,
                    IDX[name][h-1]: -1
                }, -np.inf, ramp)
                add_constraint({
                    IDX[name][h]: -1,
                    IDX[name][h-1]: 1
                }, -np.inf, ramp)

        # ----- Startup logic + minimum-on time -----
        for on, start, initial_on, min_on in [
            ("on_coal", "start_coal", 1, COAL_MIN_ON),
            ("on_bio", "start_bio", 1, BIO_MIN_ON)
        ]:
            if h == 0:
                # start >= on - initial_on
                add_constraint({
                    IDX[start][h]: 1,
                    IDX[on][h]: -1
                }, -initial_on, np.inf)

                # start <= on
                add_constraint({
                    IDX[start][h]: 1,
                    IDX[on][h]: -1
                }, -np.inf, 0)

                # initial state was already ON, so no startup at h=0
                add_constraint({
                    IDX[start][h]: 1
                }, -np.inf, 1 - initial_on)
            else:
                # start >= on[h] - on[h-1]
                add_constraint({
                    IDX[start][h]: 1,
                    IDX[on][h]: -1,
                    IDX[on][h-1]: 1
                }, 0, np.inf)

                # start <= on[h]
                add_constraint({
                    IDX[start][h]: 1,
                    IDX[on][h]: -1
                }, -np.inf, 0)

                # start <= 1 - on[h-1]
                add_constraint({
                    IDX[start][h]: 1,
                    IDX[on][h-1]: 1
                }, -np.inf, 1)

            # If a unit starts, it must remain ON for the minimum-on period.
            if h + min_on <= 24:
                coeffs = {IDX[on][j]: 1 for j in range(h, h + min_on)}
                coeffs[IDX[start][h]] = -min_on
                add_constraint(coeffs, 0, np.inf)
            else:
                # Cannot start near the end if the full minimum-on period
                # cannot fit inside the 24-hour scored horizon.
                add_constraint({IDX[start][h]: 1}, 0, 0)

    # ----- Flexible production -----
    add_constraint(
        {IDX["flex"][h]: 1 for h in range(24)},
        FLEX_DAILY_ENERGY, FLEX_DAILY_ENERGY
    )

    for h in range(CONTINUITY_START, CONTINUITY_END + 1):
        add_constraint({IDX["flex"][h]: 1}, CONTINUITY_MIN, np.inf)

    # ----- Daily fuel limits including startup fuel -----
    add_constraint(
        {IDX["coal"][h]: COAL_FUEL_COEFF for h in range(24)}
        | {IDX["start_coal"][h]: 6.0 for h in range(24)},
        -np.inf, COAL_DAILY_LIMIT
    )

    add_constraint(
        {IDX["bio"][h]: BIO_FUEL_COEFF for h in range(24)}
        | {IDX["start_bio"][h]: 5.0 for h in range(24)},
        -np.inf, BIO_DAILY_LIMIT
    )

    add_constraint(
        {IDX["gas"][h]: GAS_FUEL_COEFF for h in range(24)},
        -np.inf, GAS_DAILY_LIMIT
    )

    # ----- Emission and cost expressions -----
    emission_coeff = np.zeros(NVAR)
    emission_coeff[IDX["coal"]] = COAL_FUEL_COEFF * COAL_EF
    emission_coeff[IDX["bio"]] = BIO_FUEL_COEFF * BIO_EF
    emission_coeff[IDX["gas"]] = GAS_FUEL_COEFF * GAS_EF
    emission_coeff[IDX["grid"]] = grid_factor

    cost_coeff = np.zeros(NVAR)
    cost_coeff[IDX["coal"]] = COAL_FUEL_COEFF * COAL_PRICE
    cost_coeff[IDX["bio"]] = BIO_FUEL_COEFF * BIO_PRICE
    cost_coeff[IDX["gas"]] = GAS_FUEL_COEFF * GAS_PRICE
    cost_coeff[IDX["grid"]] = tariff

    if carbon_cap is not None:
        add_constraint(
            {i: v for i, v in enumerate(emission_coeff) if v != 0},
            -np.inf, carbon_cap
        )

    if cost_cap is not None:
        add_constraint(
            {i: v for i, v in enumerate(cost_coeff) if v != 0},
            -np.inf, cost_cap
        )

    # Primary objective = emissions.
    # Tiny normalised cost term breaks near-equal emission ties.
    objective = emission_coeff + (1e-7 * cost_coeff)

    A = lil_matrix((len(rows), NVAR))
    for r, coeffs in enumerate(rows):
        for i, value in coeffs.items():
            A[r, i] = value

    result = milp(
        objective,
        integrality=integer,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            A.tocsr(), np.asarray(lows), np.asarray(highs)
        ),
        options={"time_limit": 60}
    )

    if not result.success:
        raise RuntimeError(
            "No feasible dispatch was found. Solver message: "
            + str(result.message)
        )

    return result.x

# ============================================================
# 5. CONVERT SOLVER OUTPUT INTO AUDITABLE HOURLY TABLE
# ============================================================

def build_plan(x):
    coal = x[IDX["coal"]]
    bio = x[IDX["bio"]]
    gas = x[IDX["gas"]]
    wh = x[IDX["wh"]]
    solar = x[IDX["solar"]]
    grid = x[IDX["grid"]]
    flex = x[IDX["flex"]]

    coal_fuel = coal * COAL_FUEL_COEFF
    bio_fuel = bio * BIO_FUEL_COEFF
    gas_fuel = gas * GAS_FUEL_COEFF

    steam_supply = coal + bio + gas + wh
    electricity_demand = fixed_electricity + flex
    electricity_supply = solar + grid

    coal_emissions = coal_fuel * COAL_EF
    bio_emissions = bio_fuel * BIO_EF
    gas_emissions = gas_fuel * GAS_EF
    grid_emissions = grid * grid_factor

    emissions = (
        coal_emissions + bio_emissions
        + gas_emissions + grid_emissions
    )

    coal_cost = coal_fuel * COAL_PRICE
    bio_cost = bio_fuel * BIO_PRICE
    gas_cost = gas_fuel * GAS_PRICE
    grid_cost = grid * tariff

    cost = coal_cost + bio_cost + gas_cost + grid_cost

    return pd.DataFrame({
        "Hour": hour_labels,
        "Steam Demand (t/h)": steam_demand,
        "Coal (t/h)": coal,
        "Biomass (t/h)": bio,
        "Gas (t/h)": gas,
        "Waste Heat (t/h)": wh,
        "Steam Supply (t/h)": steam_supply,
        "Steam Error (t/h)": steam_supply - steam_demand,
        "Solar Used (MW)": solar,
        "Grid Import (MW)": grid,
        "Fixed Electricity (MW)": fixed_electricity,
        "Flexible Load (MW)": flex,
        "Electricity Demand (MW)": electricity_demand,
        "Electricity Error (MW)": electricity_supply - electricity_demand,
        "Coal Fuel (t)": coal_fuel,
        "Biomass Fuel (t)": bio_fuel,
        "Gas Fuel (Sm3)": gas_fuel,
        "Emissions (tCO2e)": emissions,
        "Variable Cost (INR)": cost
    })

# ============================================================
# 6. RUN BASELINE + NORMAL + RESILIENCE
# ============================================================

try:
    baseline = reconstruct_official_baseline()

    # Qualification gates:
    # carbon <= 88% of official baseline
    # cost <= 95% of official baseline
    normal_x = solve_dispatch(
        resilience=False,
        carbon_cap=0.88 * BASELINE_CO2,
        cost_cap=0.95 * BASELINE_COST
    )
    normal_df = build_plan(normal_x)

    resilience_x = solve_dispatch(
        resilience=True
    )
    resilience_df = build_plan(resilience_x)

except Exception as e:
    st.error(f"Solver error: {e}")
    st.stop()

# ============================================================
# 7. SUMMARY / VALIDATION FUNCTIONS
# ============================================================

def summary(df):
    return {
        "CO2": df["Emissions (tCO2e)"].sum(),
        "Cost": df["Variable Cost (INR)"].sum(),
        "Coal Fuel": df["Coal Fuel (t)"].sum(),
        "Biomass Fuel": df["Biomass Fuel (t)"].sum(),
        "Gas Fuel": df["Gas Fuel (Sm3)"].sum(),
        "Grid": df["Grid Import (MW)"].sum(),
        "Solar": df["Solar Used (MW)"].sum(),
        "Flexible": df["Flexible Load (MW)"].sum(),
        "Max Grid": df["Grid Import (MW)"].max(),
        "Max Coal": df["Coal (t/h)"].max(),
        "Max Biomass": df["Biomass (t/h)"].max(),
        "Max Gas": df["Gas (t/h)"].max(),
        "Max Steam Over": (
            df["Steam Supply (t/h)"]
            / df["Steam Demand (t/h)"]
        ).max()
    }

normal = summary(normal_df)
resilience = summary(resilience_df)

def check_plan(df, resilience=False):
    checks = []

    checks.append((
        "Hourly steam balance lower bound",
        np.min(
            df["Steam Supply (t/h)"] - df["Steam Demand (t/h)"]
        ) >= -1e-6
    ))

    checks.append((
        "Hourly steam balance upper bound <= 103%",
        np.max(
            df["Steam Supply (t/h)"]
            - 1.03 * df["Steam Demand (t/h)"]
        ) <= 1e-6
    ))

    checks.append((
        "Hourly electricity balance",
        np.max(np.abs(
            df["Electricity Error (MW)"]
        )) <= 1e-6
    ))

    checks.append((
        "Coal fuel <= 350 t/day",
        normal["Coal Fuel"] <= COAL_DAILY_LIMIT + 1e-6
        if df is normal_df else
        df["Coal Fuel (t)"].sum() <= COAL_DAILY_LIMIT + 1e-6
    ))

    checks.append((
        "Biomass fuel <= 220 t/day",
        df["Biomass Fuel (t)"].sum() <= BIO_DAILY_LIMIT + 1e-6
    ))

    checks.append((
        "Gas fuel <= 20,000 Sm3/day",
        df["Gas Fuel (Sm3)"].sum() <= GAS_DAILY_LIMIT + 1e-6
    ))

    checks.append((
        "Grid import <= 22 MW",
        df["Grid Import (MW)"].max() <= GRID_LIMIT + 1e-6
    ))

    checks.append((
        "Flexible production = 18 MWh/day",
        abs(df["Flexible Load (MW)"].sum() - 18.0) <= 1e-6
    ))

    checks.append((
        "Flexible load continuity 08:00-18:00",
        df.loc[8:18, "Flexible Load (MW)"].min() >= 0.5 - 1e-6
    ))

    if resilience:
        checks.append((
            "Resilience biomass <= 20 t/h at 14:00-18:00",
            df.loc[14:17, "Biomass (t/h)"].max() <= 20.0 + 1e-6
        ))

    return checks

normal_checks = check_plan(normal_df)
resilience_checks = check_plan(resilience_df, resilience=True)


# ============================================================
# 90-DAY OPERATIONAL PLAN MODE
# ============================================================

def render_90_day_mode():
    st.title("📅 CarbonOS — 90-Day Operational Plan")
    st.subheader("Daily Demand → Daily Simulation → 90-Day Evidence")

    # Keep the 90-day data in Streamlit session memory.
    # No database is required; the user can download the CSV at any time.
    if "plan90" not in st.session_state:
        st.session_state.plan90 = {}

    st.info(
        "Enter the plant demand for one day, save it, and repeat through Day 90. "
        "The downloaded CSV is the auditable record of the 90-day plan."
    )

    # 90-day targets based on the supplied project targets.
    ANNUAL_CO2_BASELINE = 296636.0
    ANNUAL_CO2_REDUCTION_TARGET = 59327.0       # 20%
    ANNUAL_WATER_BASELINE = 2.75               # million m³/year
    ANNUAL_WATER_REDUCTION_TARGET = 0.275      # million m³/year
    DAILY_CO2_BASELINE = ANNUAL_CO2_BASELINE / 365.0
    DAILY_CO2_TARGET = DAILY_CO2_BASELINE * 0.80
    DAILY_WATER_BASELINE = ANNUAL_WATER_BASELINE / 365.0
    DAILY_WATER_TARGET = DAILY_WATER_BASELINE * 0.90

    day90 = st.number_input(
        "Select planning day",
        min_value=1,
        max_value=90,
        value=1,
        step=1,
        key="plan90_day"
    )

    st.markdown(f"### Day {day90} — Enter Demand")

    c1, c2, c3 = st.columns(3)

    with c1:
        daily_steam = st.number_input(
            "Steam demand (t/day)",
            min_value=0.0,
            value=float(steam_demand.sum()),
            step=10.0,
            key="p90_steam"
        )
        daily_coal = st.number_input(
            "Coal consumption (t/day)",
            min_value=0.0,
            value=100.0,
            step=5.0,
            key="p90_coal"
        )
        daily_biomass = st.number_input(
            "Biomass consumption (t/day)",
            min_value=0.0,
            value=50.0,
            step=5.0,
            key="p90_bio"
        )

    with c2:
        daily_gas = st.number_input(
            "Natural gas (Sm³/day)",
            min_value=0.0,
            value=5000.0,
            step=500.0,
            key="p90_gas"
        )
        daily_grid = st.number_input(
            "Grid electricity (MWh/day)",
            min_value=0.0,
            value=float(fixed_electricity.sum()),
            step=5.0,
            key="p90_grid"
        )
        daily_water = st.number_input(
            "Freshwater demand (million m³/day)",
            min_value=0.0,
            value=float(DAILY_WATER_BASELINE),
            step=0.0005,
            format="%.4f",
            key="p90_water"
        )

    with c3:
        production90 = st.number_input(
            "Production achieved (%)",
            min_value=0.0,
            max_value=100.0,
            value=99.0,
            step=0.5,
            key="p90_production"
        )
        v2v_active = st.checkbox(
            "Activate V2V for this day",
            value=False,
            key="p90_v2v"
        )
        v2v_coal_saved = st.number_input(
            "V2V coal displaced (t/day)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="p90_v2v_coal"
        ) if v2v_active else 0.0

        v2v_water = st.number_input(
            "V2V water recovered (m³/day)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            key="p90_v2v_water"
        ) if v2v_active else 0.0

    # Use the same emission factors already defined in this simulator.
    coal_co2 = daily_coal * COAL_EF
    biomass_co2 = daily_biomass * BIO_EF
    gas_co2 = daily_gas * GAS_EF
    grid_co2 = daily_grid * grid_factor.mean()

    v2v_co2_avoided = v2v_coal_saved * COAL_EF
    gross_co2 = coal_co2 + biomass_co2 + gas_co2 + grid_co2
    net_co2 = max(0.0, gross_co2 - v2v_co2_avoided)

    recovered_m3 = v2v_water
    net_water = max(0.0, daily_water - recovered_m3 / 1_000_000)

    carbon_reduction_pct = (
        100.0 * (DAILY_CO2_BASELINE - net_co2) / DAILY_CO2_BASELINE
    )
    water_reduction_pct = (
        100.0 * (DAILY_WATER_BASELINE - net_water) / DAILY_WATER_BASELINE
    )

    st.divider()
    st.subheader("📊 Day Result")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net CO₂", f"{net_co2:,.1f} tCO₂e")
    k2.metric("CO₂ Reduction", f"{carbon_reduction_pct:.1f}%")
    k3.metric("Net Freshwater", f"{net_water:.4f} M m³")
    k4.metric("Production", f"{production90:.1f}%")

    carbon_ok = net_co2 <= DAILY_CO2_TARGET
    production_ok = production90 >= 98.0

    s1, s2 = st.columns(2)
    with s1:
        if carbon_ok:
            st.success("✅ Day meets the 20% carbon-reduction target.")
        else:
            st.warning("⚠️ Day is above the 20% carbon-reduction target.")

    with s2:
        if production_ok:
            st.success("✅ Production requirement ≥98% maintained.")
        else:
            st.error("❌ Production is below the required 98%.")

    if v2v_active:
        st.success(
            f"♻️ V2V contribution: {v2v_co2_avoided:.1f} tCO₂e avoided + "
            f"{recovered_m3:,.0f} m³ water recovered."
        )

    if st.button(f"💾 Save Day {day90}", type="primary", key="save_p90"):
        st.session_state.plan90[day90] = {
            "Day": day90,
            "Steam Demand (t/day)": daily_steam,
            "Coal (t/day)": daily_coal,
            "Biomass (t/day)": daily_biomass,
            "Natural Gas (Sm3/day)": daily_gas,
            "Grid Electricity (MWh/day)": daily_grid,
            "Freshwater Demand (M m3/day)": daily_water,
            "V2V Active": "Yes" if v2v_active else "No",
            "V2V Coal Saved (t/day)": v2v_coal_saved,
            "V2V CO2 Avoided (tCO2e)": v2v_co2_avoided,
            "V2V Water Recovered (m3/day)": recovered_m3,
            "Net CO2 (tCO2e/day)": net_co2,
            "CO2 Reduction (%)": carbon_reduction_pct,
            "Net Freshwater (M m3/day)": net_water,
            "Water Reduction (%)": water_reduction_pct,
            "Production (%)": production90,
            "Carbon Target Met": "Yes" if carbon_ok else "No",
            "Production Target Met": "Yes" if production_ok else "No",
        }
        st.success(f"Day {day90} saved. You can now move to Day {min(day90 + 1, 90)}.")

    # --------------------------------------------------------
    # 90-DAY DASHBOARD
    # --------------------------------------------------------
    st.divider()
    st.subheader("📈 90-Day Progress")

    if st.session_state.plan90:
        plan_df = pd.DataFrame(
            list(st.session_state.plan90.values())
        ).sort_values("Day")

        completed = len(plan_df)
        days_progress = completed / 90.0

        st.progress(days_progress)
        st.write(f"**{completed}/90 days completed ({days_progress * 100:.1f}%)**")

        total_v2v_co2 = plan_df["V2V CO2 Avoided (tCO2e)"].sum()
        total_water_recovery = plan_df["V2V Water Recovered (m3/day)"].sum() / 1_000_000
        avg_production = plan_df["Production (%)"].mean()

        # Equivalent 90-day target.
        target_90_co2_avoided = ANNUAL_CO2_REDUCTION_TARGET * 90 / 365
        target_90_water_saved = ANNUAL_WATER_REDUCTION_TARGET * 90 / 365

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("V2V CO₂ Avoided", f"{total_v2v_co2:,.1f} t")
        c2.metric("V2V Water Recovered", f"{total_water_recovery:.4f} M m³")
        c3.metric("Avg Production", f"{avg_production:.1f}%")
        c4.metric(
            "Target Days",
            f"{int((plan_df['Carbon Target Met'] == 'Yes').sum())}/{completed}"
        )

        st.write("**Carbon reduction target progress**")
        carbon_progress = min(
            total_v2v_co2 / target_90_co2_avoided
            if target_90_co2_avoided > 0 else 0.0,
            1.0
        )
        st.progress(carbon_progress)
        st.caption(
            f"V2V-only contribution: {total_v2v_co2:,.1f} / "
            f"{target_90_co2_avoided:,.1f} tCO₂e 90-day target."
        )

        st.write("**Water recovery target progress**")
        water_progress = min(
            total_water_recovery / target_90_water_saved
            if target_90_water_saved > 0 else 0.0,
            1.0
        )
        st.progress(water_progress)
        st.caption(
            f"V2V water recovery: {total_water_recovery:.4f} / "
            f"{target_90_water_saved:.4f} M m³ 90-day target."
        )

        st.subheader("📋 Saved 90-Day Plan")
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        # Download complete plan.
        csv90 = plan_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Complete 90-Day Plan",
            data=csv90,
            file_name="CarbonOS_90_Day_Operational_Plan.csv",
            mime="text/csv",
            key="download_90_plan"
        )

        # Download selected day.
        if day90 in st.session_state.plan90:
            day_csv = pd.DataFrame(
                [st.session_state.plan90[day90]]
            ).to_csv(index=False).encode("utf-8")

            st.download_button(
                f"⬇️ Download Day {day90} Data",
                data=day_csv,
                file_name=f"CarbonOS_Day_{day90}.csv",
                mime="text/csv",
                key="download_selected_day"
            )
    else:
        st.info("No days saved yet. Save Day 1 to start building the 90-day evidence file.")

    st.divider()
    st.caption(
        "90-Day mode is an operational planning/evidence layer. "
        "Normal Operation and Resilience Mode remain unchanged."
    )


# ============================================================
# MODE SELECTOR
# ============================================================

st.sidebar.header("⚙️ CarbonOS Mode")
mode = st.sidebar.radio(
    "Select mode:",
    ["Normal Operation", "Resilience Mode", "90-Day Operational Plan"],
    key="carbonos_mode"
)

if mode == "90-Day Operational Plan":
    render_90_day_mode()
    st.stop()

# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.header("⚙️ CarbonOS Scenario")
# Normal/resilience selection is used only after 90-Day Operational Plan
# has been handled above.
mode = st.sidebar.radio(
    "Select scenario:",
    ["Normal Operation", "Resilience Mode"],
    key="scenario_mode"
)

active_df = normal_df if mode == "Normal Operation" else resilience_df
active = normal if mode == "Normal Operation" else resilience
active_checks = normal_checks if mode == "Normal Operation" else resilience_checks

if mode == "Normal Operation":
    st.success("🟢 CARBONOS: NORMAL OPERATION")
else:
    st.warning("🟠 CARBONOS: MANDATORY RESILIENCE MODE")

# ============================================================
# 9. PERFORMANCE
# ============================================================

carbon_improvement = 100 * (BASELINE_CO2 - active["CO2"]) / BASELINE_CO2
cost_improvement = 100 * (BASELINE_COST - active["Cost"]) / BASELINE_COST

c1, c2, c3, c4 = st.columns(4)
c1.metric("24-h CO₂", f"{active['CO2']:.2f} tCO₂e")
c2.metric("24-h Variable Cost", f"₹{active['Cost']:,.0f}")
c3.metric("CO₂ vs Official Baseline", f"{carbon_improvement:.2f}%")
c4.metric("Cost vs Official Baseline", f"{cost_improvement:.2f}%")

if mode == "Normal Operation":
    gate1 = active["CO2"] <= 0.88 * BASELINE_CO2 + 1e-6
    gate2 = active["Cost"] <= 0.95 * BASELINE_COST + 1e-6
    if gate1 and gate2:
        st.success("✅ Normal-case carbon and variable-cost qualification gates are satisfied.")
    else:
        st.error("❌ A normal-case qualification gate is not satisfied.")

st.divider()

# ============================================================
# 10. OFFICIAL BASELINE RECONSTRUCTION
# ============================================================

st.subheader("📌 Official Baseline Reconstruction")

baseline_totals = {
    "Steam Demand": steam_demand.sum(),
    "Coal Steam": baseline["coal"].sum(),
    "Biomass Steam": baseline["bio"].sum(),
    "Gas Steam": baseline["gas"].sum(),
    "Waste Heat Steam": baseline["wh"].sum(),
    "Grid Import": baseline["grid"].sum(),
    "Solar Used": baseline["solar"].sum(),
    "CO2": baseline["emissions"].sum(),
    "Cost": baseline["cost"].sum()
}

baseline_table = pd.DataFrame({
    "Metric": [
        "Steam demand",
        "Coal steam",
        "Biomass steam",
        "Gas steam",
        "Waste-heat steam",
        "Grid import",
        "Solar used",
        "Challenge emissions",
        "Variable cost"
    ],
    "Reconstructed": [
        baseline_totals["Steam Demand"],
        baseline_totals["Coal Steam"],
        baseline_totals["Biomass Steam"],
        baseline_totals["Gas Steam"],
        baseline_totals["Waste Heat Steam"],
        baseline_totals["Grid Import"],
        baseline_totals["Solar Used"],
        baseline_totals["CO2"],
        baseline_totals["Cost"]
    ],
    "Official": [
        2797.0, 1886.0, 480.0, 205.0, 226.0,
        216.5, 59.2, 986.992, 5526720
    ]
})
baseline_table["Difference"] = (
    baseline_table["Reconstructed"] - baseline_table["Official"]
)
st.dataframe(baseline_table, use_container_width=True, hide_index=True)

# ============================================================
# 11. HOURLY DECISION
# ============================================================

st.subheader("⏱️ Hourly CarbonOS Decision")
h = st.slider("Select hour", 0, 23, 12)

r = active_df.iloc[h]
a, b, c = st.columns(3)
a.metric("Steam Demand", f"{r['Steam Demand (t/h)']:.1f} t/h")
a.metric("Steam Supply", f"{r['Steam Supply (t/h)']:.1f} t/h")

b.metric("Solar Used", f"{r['Solar Used (MW)']:.2f} MW")
b.metric("Grid Import", f"{r['Grid Import (MW)']:.2f} MW")

c.metric("CO₂", f"{r['Emissions (tCO2e)']:.3f} t")
c.metric("Variable Cost", f"₹{r['Variable Cost (INR)']:,.0f}")

st.write(f"### {hour_labels[h]}")

dispatch = pd.DataFrame({
    "Source": ["Coal", "Biomass", "Gas", "Waste Heat"],
    "Setpoint (t/h)": [
        r["Coal (t/h)"],
        r["Biomass (t/h)"],
        r["Gas (t/h)"],
        r["Waste Heat (t/h)"]
    ]
})
st.table(dispatch)

if abs(r["Steam Error (t/h)"]) <= 1e-6:
    st.success("✓ Steam balance exactly closed.")
else:
    st.info(
        f"Steam is intentionally allowed up to 103% by the problem statement. "
        f"Current error = {r['Steam Error (t/h)']:.3f} t/h."
    )

if abs(r["Electricity Error (MW)"]) <= 1e-6:
    st.success("✓ Electricity balance closed.")
else:
    st.error("✗ Electricity balance failed.")

# ============================================================
# 12. FULL HOURLY PLAN
# ============================================================

st.subheader("📋 Full 24-Hour Dispatch Plan")
st.dataframe(active_df, use_container_width=True, hide_index=True)

# ============================================================
# 13. STEAM GRAPH
# ============================================================

st.subheader("🔥 24-Hour Steam Dispatch")
fig1 = plt.figure(figsize=(12, 5))
plt.stackplot(
    hours,
    active_df["Coal (t/h)"],
    active_df["Biomass (t/h)"],
    active_df["Gas (t/h)"],
    active_df["Waste Heat (t/h)"],
    labels=["Coal", "Biomass", "Gas", "Waste Heat"]
)
plt.plot(
    hours,
    active_df["Steam Demand (t/h)"],
    "--",
    linewidth=2,
    label="Steam Demand"
)
plt.xticks(hours)
plt.xlabel("Hour")
plt.ylabel("Steam (t/h)")
plt.title("CarbonOS Steam Dispatch")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
st.pyplot(fig1)

# ============================================================
# 14. ELECTRICITY GRAPH
# ============================================================

st.subheader("⚡ Electricity Balance")
fig2 = plt.figure(figsize=(12, 5))
plt.plot(
    hours,
    active_df["Solar Used (MW)"],
    marker="o",
    label="Solar Used"
)
plt.plot(
    hours,
    active_df["Grid Import (MW)"],
    marker="s",
    label="Grid Import"
)
plt.plot(
    hours,
    active_df["Electricity Demand (MW)"],
    "--",
    linewidth=2,
    label="Total Demand"
)
plt.xticks(hours)
plt.xlabel("Hour")
plt.ylabel("Power (MW)")
plt.title("CarbonOS Electricity Dispatch")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
st.pyplot(fig2)

# ============================================================
# 15. CONSTRAINT DASHBOARD
# ============================================================

st.subheader("🛡️ Constraint Dashboard")

constraint_rows = []
for name, passed in active_checks:
    constraint_rows.append([
        name,
        "PASS ✓" if passed else "FAIL ✗"
    ])

constraint_table = pd.DataFrame(
    constraint_rows,
    columns=["Constraint", "Status"]
)
st.dataframe(
    constraint_table,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# 16. FUEL SUMMARY
# ============================================================

st.subheader("⛽ 24-Hour Fuel Summary")
fuel_table = pd.DataFrame({
    "Fuel": ["Coal", "Eligible Biomass", "Natural Gas"],
    "Consumption": [
        f"{active['Coal Fuel']:.2f} t",
        f"{active['Biomass Fuel']:.2f} t",
        f"{active['Gas Fuel']:.0f} Sm3"
    ],
    "Limit": [
        "350 t/day",
        "220 t/day",
        "20,000 Sm3/day"
    ]
})
st.table(fuel_table)

# ============================================================
# 17. NORMAL VS RESILIENCE
# ============================================================

st.subheader("🔄 Normal vs Mandatory Resilience")

comparison = pd.DataFrame({
    "Scenario": ["Official Baseline", "Normal", "Resilience"],
    "CO₂ (tCO₂e)": [
        BASELINE_CO2,
        normal["CO2"],
        resilience["CO2"]
    ],
    "Variable Cost (INR)": [
        BASELINE_COST,
        normal["Cost"],
        resilience["Cost"]
    ],
    "Grid (MWh)": [
        baseline["grid"].sum(),
        normal["Grid"],
        resilience["Grid"]
    ],
    "Solar Used (MWh)": [
        baseline["solar"].sum(),
        normal["Solar"],
        resilience["Solar"]
    ]
})
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.write(
    f"**Resilience change vs team's normal plan:** "
    f"{resilience['CO2'] - normal['CO2']:+.2f} tCO₂e, "
    f"₹{resilience['Cost'] - normal['Cost']:+,.0f} variable cost."
)

# ============================================================
# 18. DOWNLOAD
# ============================================================

st.subheader("📥 Simulation Data")
csv_data = active_df.to_csv(index=False)
st.download_button(
    "Download 24-hour CarbonOS CSV",
    csv_data,
    "CarbonOS_24h_dispatch.csv",
    "text/csv"
)

# ============================================================
# 19. FINAL STATUS
# ============================================================

all_pass = all(passed for _, passed in active_checks)

st.divider()

if all_pass:
    st.success(
        "🟢 CARBONOS DECISION: FEASIBLE — "
        "all checked hard constraints are satisfied."
    )
else:
    st.error(
        "🔴 CARBONOS DECISION: INFEASIBLE — "
        "at least one hard constraint is violated."
    )

st.caption(
    "This simulator uses the complete 24-hour Table 5 inputs and "
    "the hard constraints specified in DECARBONIX 1.0 Problem 02. "
    "The dispatch is generated by a transparent MILP rather than "
    "hard-coded hourly set-points."
)

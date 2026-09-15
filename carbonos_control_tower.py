

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CARBONOS CONTROL TOWER
# DECARBONIX 1.0 - Problem 02
# ============================================================

st.set_page_config(
    page_title="CarbonOS Control Tower",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 CarbonOS — Control Tower")
st.subheader("From Target → Feasible Shift Plan")

st.markdown(
    """
    **CarbonOS** is an hourly dispatch decision-support prototype.
    
    It checks:
    - Steam balance
    - Electricity balance
    - Fuel consumption
    - Carbon emissions
    - Variable cost
    - Operating constraints
    - Resilience under biomass and solar reduction
    """
)

# ============================================================
# 1. HOURLY PLANT INPUT DATA
# ============================================================

hours = list(range(24))

steam_demand = np.array([
    92,88,86,84,86,92,100,110,122,132,140,146,
    150,148,145,142,138,134,128,120,112,106,100,96
])

waste_heat_avail = np.array([
    8,8,8,8,8,10,12,15,18,20,20,20,
    20,20,20,18,18,16,14,12,10,10,8,8
])

fixed_electricity = np.array([
    8.5,8.2,8.0,8.1,8.4,9.0,9.8,10.5,
    11.2,11.8,12.4,12.8,13.0,13.2,13.0,
    12.7,12.5,12.2,11.8,11.5,10.8,10.0,9.4,8.9
])

solar_availability = np.array([
    0,0,0,0,0,0,0.5,2,4,6,8,9.5,
    10,9.5,8.5,7,5,3,1,0,0,0,0,0
])

grid_factor = np.array([
    0.78,0.76,0.74,0.72,0.70,0.68,
    0.66,0.64,0.60,0.56,0.52,0.48,
    0.46,0.45,0.47,0.50,0.55,0.62,
    0.70,0.78,0.82,0.84,0.82,0.80
])

tariff = np.array([
    5200,5200,5200,5200,5200,5200,
    6800,6800,6800,6800,
    5600,5600,5600,5600,5600,5600,
    7200,7200,7200,
    8200,8200,8200,8200,
    6000
])

# ============================================================
# 2. NORMAL CARBONOS DISPATCH
# ============================================================

coal_normal = np.array([
    64,60,58,56,50,60,68,67,57,63,53,63,
    73,79,69,64,74,80,84,86,76,76,66,56
])

biomass_normal = np.array([
    20,20,20,20,28,22,20,28,36,44,52,60,
    57,49,45,53,46,38,30,22,26,20,26,32
])

gas_normal = np.array([
    0,0,0,0,0,0,
    0,0,11,5,15,3,
    0,0,11,7,0,0,
    0,0,0,0,0,0
])

waste_heat_normal = np.array([
    8,8,8,8,8,10,12,15,18,20,20,20,
    20,20,20,18,18,16,14,12,10,10,8,8
])

solar_normal = np.array([
    0,0,0,0,0,0,
    0.5,2,4,6,8,9.5,
    10,9.5,8.5,7,5,3,1,0,0,0,0,0
])

flexible_load = np.array([
    0,0,0,0,0,0,
    0,0,0.5,0.5,
    2.5,2.5,2.5,2.5,2.5,2.5,
    1.5,0.5,
    0,0,0,0,0,0
])

# ============================================================
# 3. FUEL / COST / EMISSION PARAMETERS
# ============================================================

coal_fuel_coeff = 0.18
biomass_fuel_coeff = 0.27
gas_fuel_coeff = 80

coal_emission_factor = 2.35
biomass_emission_factor = 0.10
gas_emission_factor = 0.00200

coal_price = 8500
biomass_price = 5500
gas_price = 32

coal_daily_limit = 350
biomass_daily_limit = 220
gas_daily_limit = 20000

grid_limit = 22

# ============================================================
# 4. FUNCTION TO BUILD A DISPATCH PLAN
# ============================================================

def build_plan(
    coal,
    biomass,
    gas,
    waste_heat,
    solar,
    flexible
):

    # Electricity required from grid
    grid = fixed_electricity + flexible - solar

    # Prevent negative grid import
    grid = np.maximum(grid, 0)

    # Fuel consumption
    coal_fuel = coal * coal_fuel_coeff
    biomass_fuel = biomass * biomass_fuel_coeff
    gas_fuel = gas * gas_fuel_coeff

    # Emissions
    coal_emissions = coal_fuel * coal_emission_factor
    biomass_emissions = biomass_fuel * biomass_emission_factor
    gas_emissions = gas_fuel * gas_emission_factor
    grid_emissions = grid * grid_factor

    total_emissions_hourly = (
        coal_emissions
        + biomass_emissions
        + gas_emissions
        + grid_emissions
    )

    # Cost
    coal_cost = coal_fuel * coal_price
    biomass_cost = biomass_fuel * biomass_price
    gas_cost = gas_fuel * gas_price
    grid_cost = grid * tariff

    total_cost_hourly = (
        coal_cost
        + biomass_cost
        + gas_cost
        + grid_cost
    )

    # Steam balance
    steam_supply = (
        coal
        + biomass
        + gas
        + waste_heat
    )

    steam_error = steam_supply - steam_demand

    # Electricity balance
    electricity_supply = solar + grid
    electricity_demand = fixed_electricity + flexible

    electricity_error = (
        electricity_supply - electricity_demand
    )

    df = pd.DataFrame({
        "Hour": hours,
        "Steam Demand": steam_demand,
        "Coal Steam": coal,
        "Biomass Steam": biomass,
        "Gas Steam": gas,
        "Waste Heat": waste_heat,
        "Steam Supply": steam_supply,
        "Steam Error": steam_error,
        "Solar": solar,
        "Grid": grid,
        "Fixed Electricity": fixed_electricity,
        "Flexible Load": flexible,
        "Electricity Error": electricity_error,
        "Coal Fuel": coal_fuel,
        "Biomass Fuel": biomass_fuel,
        "Gas Fuel": gas_fuel,
        "Emissions": total_emissions_hourly,
        "Cost": total_cost_hourly
    })

    return df


# ============================================================
# 5. BUILD NORMAL PLAN
# ============================================================

normal_df = build_plan(
    coal_normal,
    biomass_normal,
    gas_normal,
    waste_heat_normal,
    solar_normal,
    flexible_load
)
# ============================================================
# 6. RESILIENCE SCENARIO
# ============================================================

coal_resilience = coal_normal.copy()
biomass_resilience = biomass_normal.copy()
gas_resilience = gas_normal.copy()
waste_heat_resilience = waste_heat_normal.copy()
solar_resilience = solar_normal.copy()

# Resilience dispatch adjustments
coal_resilience[11:19] = [
    63,73,83,90,90,90,88,82
]

biomass_resilience[11:19] = [
    44,36,28,20,20,20,20,28
]

gas_resilience[11:19] = [
    19,21,17,15,14,10,10,4
]

# Solar reduction during resilience
for h in range(12,16):
    solar_resilience[h] = (
        0.60 * solar_normal[h]
    )

# ============================================================
# 7. BUILD RESILIENCE PLAN
# ============================================================

resilience_df = build_plan(
    coal_resilience,
    biomass_resilience,
    gas_resilience,
    waste_heat_resilience,
    solar_resilience,
    flexible_load
)

# ============================================================
# 8. BASELINE
# ============================================================

BASELINE_CO2 = 986.992
BASELINE_COST = 5526720

# ============================================================
# 9. SUMMARY FUNCTION
# ============================================================

def calculate_summary(df):

    return {
        "CO2": df["Emissions"].sum(),
        "Cost": df["Cost"].sum(),
        "Coal Fuel": df["Coal Fuel"].sum(),
        "Biomass Fuel": df["Biomass Fuel"].sum(),
        "Gas Fuel": df["Gas Fuel"].sum(),
        "Grid": df["Grid"].sum(),
        "Solar": df["Solar"].sum(),
        "Max Grid": df["Grid"].max(),
        "Max Coal": df["Coal Steam"].max(),
        "Max Biomass": df["Biomass Steam"].max(),
        "Max Gas": df["Gas Steam"].max()
    }

normal_summary = calculate_summary(normal_df)

resilience_summary = calculate_summary(
    resilience_df
)

# ============================================================
# 10. MODE SELECTION
# ============================================================

st.sidebar.header("⚙️ CarbonOS Mode")

mode = st.sidebar.radio(
    "Select scenario:",
    ["Normal Operation", "Resilience Mode"]
)

if mode == "Normal Operation":

    active_df = normal_df
    active_summary = normal_summary

    st.success(
        "🟢 CARBONOS: NORMAL OPERATION"
    )

else:

    active_df = resilience_df
    active_summary = resilience_summary

    st.warning(
        "🟠 CARBONOS: RESILIENCE MODE"
   )
    # ============================================================
# 11. PERFORMANCE CALCULATIONS
# ============================================================

carbon_saving = (
    BASELINE_CO2
    - active_summary["CO2"]
)

carbon_saving_percent = (
    carbon_saving
    / BASELINE_CO2
    * 100
)

cost_saving = (
    BASELINE_COST
    - active_summary["Cost"]
)

cost_saving_percent = (
    cost_saving
    / BASELINE_COST
    * 100
)

# ============================================================
# 12. CONSTRAINT CHECK
# ============================================================

def check_constraints(df):

    checks = []

    checks.append([
        "Coal fuel",
        df["Coal Fuel"].sum(),
        350,
        "t/day"
    ])

    checks.append([
        "Biomass fuel",
        df["Biomass Fuel"].sum(),
        220,
        "t/day"
    ])

    checks.append([
        "Gas fuel",
        df["Gas Fuel"].sum(),
        20000,
        "Sm³/day"
    ])

    checks.append([
        "Maximum coal",
        df["Coal Steam"].max(),
        90,
        "t/h"
    ])

    checks.append([
        "Maximum biomass",
        df["Biomass Steam"].max(),
        65,
        "t/h"
    ])

    checks.append([
        "Maximum gas",
        df["Gas Steam"].max(),
        45,
        "t/h"
    ])

    checks.append([
        "Maximum grid",
        df["Grid"].max(),
        22,
        "MW"
    ])

    return checks


checks = check_constraints(active_df)

# ============================================================
# 13. DASHBOARD
# ============================================================

st.subheader("📊 CarbonOS Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Daily CO₂",
    f"{active_summary['CO2']:.2f} t"
)

c2.metric(
    "Daily Cost",
    f"₹{active_summary['Cost']:,.0f}"
)

c3.metric(
    "CO₂ Improvement",
    f"{carbon_saving_percent:.2f}%"
)

c4.metric(
    "Cost Improvement",
    f"{cost_saving_percent:.2f}%"
)

st.divider()

# ============================================================
# 14. HOURLY DECISION
# ============================================================

st.subheader("⏱️ Hourly CarbonOS Decision")

h = st.slider(
    "Select hour",
    0,
    23,
    9
)

st.write(
    f"### Hour: {h:02d}:00 – {(h+1)%24:02d}:00"
)

a, b, c = st.columns(3)

a.metric(
    "Steam Demand",
    f"{active_df['Steam Demand'][h]:.1f} t/h"
)

a.metric(
    "Steam Supply",
    f"{active_df['Steam Supply'][h]:.1f} t/h"
)

b.metric(
    "Solar",
    f"{active_df['Solar'][h]:.1f} MW"
)

b.metric(
    "Grid",
    f"{active_df['Grid'][h]:.1f} MW"
)

c.metric(
    "CO₂",
    f"{active_df['Emissions'][h]:.3f} t"
)

c.metric(
    "Cost",
    f"₹{active_df['Cost'][h]:,.0f}"
)

# ============================================================
# 15. HOURLY DISPATCH TABLE
# ============================================================

dispatch = pd.DataFrame({

    "Source": [
        "Coal",
        "Biomass",
        "Gas",
        "Waste Heat"
    ],

    "Setpoint (t/h)": [
        active_df["Coal Steam"][h],
        active_df["Biomass Steam"][h],
        active_df["Gas Steam"][h],
        active_df["Waste Heat"][h]
    ]
})

st.table(dispatch)
# ============================================================
# 16. BALANCE CHECK
# ============================================================

steam_ok = abs(
    active_df["Steam Error"][h]
) < 0.001

electricity_ok = abs(
    active_df["Electricity Error"][h]
) < 0.001

if steam_ok:
    st.success("✓ Steam balance CLOSED")
else:
    st.error("✗ Steam balance FAILED")

if electricity_ok:
    st.success("✓ Electricity balance CLOSED")
else:
    st.error("✗ Electricity balance FAILED")

st.divider()

# ============================================================
# 17. STEAM DISPATCH GRAPH
# ============================================================

st.subheader("🔥 24-Hour Steam Dispatch")

fig1 = plt.figure(figsize=(11,5))

plt.stackplot(
    hours,
    active_df["Coal Steam"],
    active_df["Biomass Steam"],
    active_df["Gas Steam"],
    active_df["Waste Heat"],
    labels=[
        "Coal",
        "Biomass",
        "Gas",
        "Waste Heat"
    ]
)

plt.plot(
    hours,
    active_df["Steam Demand"],
    "--",
    linewidth=2,
    label="Steam Demand"
)

plt.xlabel("Hour")
plt.ylabel("Steam (t/h)")
plt.title("CarbonOS Steam Dispatch")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

st.pyplot(fig1)

# ============================================================
# 18. ELECTRICITY GRAPH
# ============================================================

st.subheader("⚡ Electricity Balance")

fig2 = plt.figure(figsize=(11,5))

plt.plot(
    hours,
    active_df["Solar"],
    marker="o",
    label="Solar"
)

plt.plot(
    hours,
    active_df["Grid"],
    marker="s",
    label="Grid"
)

plt.plot(
    hours,
    active_df["Fixed Electricity"]
    + active_df["Flexible Load"],
    "--",
    linewidth=2,
    label="Total Demand"
)

plt.xlabel("Hour")
plt.ylabel("Power (MW)")
plt.title("CarbonOS Electricity Dispatch")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

st.pyplot(fig2)

# ============================================================
# 19. CONSTRAINT DASHBOARD
# ============================================================

st.subheader("🛡️ Constraint Dashboard")

rows = []

for name, actual, limit, unit in checks:

    utilisation = (
        actual / limit * 100
    )

    status = (
        "PASS ✓"
        if actual <= limit
        else
        "FAIL ✗"
    )

    rows.append([
        name,
        f"{actual:.2f} {unit}",
        f"{limit:.2f} {unit}",
        f"{utilisation:.1f}%",
        status
    ])

constraint_table = pd.DataFrame(
    rows,
    columns=[
        "Constraint",
        "Actual",
        "Limit",
        "Utilisation",
        "Status"
    ]
)

st.dataframe(
    constraint_table,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# 20. NORMAL VS RESILIENCE
# ============================================================

st.subheader("🔄 Normal vs Resilience")

comparison = pd.DataFrame({

    "Scenario": [
        "Normal",
        "Resilience"
    ],

    "CO₂ (tCO₂e)": [
        normal_summary["CO2"],
        resilience_summary["CO2"]
    ],

    "Cost (₹)": [
        normal_summary["Cost"],
        resilience_summary["Cost"]
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# 21. FUEL SUMMARY
# ============================================================

st.subheader("⛽ Daily Fuel Summary")

fuel_table = pd.DataFrame({

    "Fuel": [
        "Coal",
        "Biomass",
        "Gas"
    ],

    "Consumption": [
        f"{active_summary['Coal Fuel']:.2f} t",
        f"{active_summary['Biomass Fuel']:.2f} t",
        f"{active_summary['Gas Fuel']:.0f} Sm³"
    ],

    "Limit": [
        "350 t/day",
        "220 t/day",
        "20,000 Sm³/day"
    ]
})

st.table(fuel_table)

# ============================================================
# 22. DOWNLOAD DATA
# ============================================================

st.subheader("📥 Simulation Data")

csv_data = active_df.to_csv(
    index=False
)

st.download_button(
    "Download Hourly CarbonOS Data",
    csv_data,
    "CarbonOS_hourly_simulation.csv",
    "text/csv"
)

# ============================================================
# 23. FINAL STATUS
# ============================================================

st.divider()

all_constraints_pass = all(
    actual <= limit
    for name, actual, limit, unit in checks
)

if all_constraints_pass:

    st.success(
        "🟢 CARBONOS DECISION: FEASIBLE — "
        "All operating constraints satisfied."
    )

else:

    st.error(
        "🔴 CARBONOS DECISION: INFEASIBLE — "
        "At least one constraint is violated."
    )

st.caption(
    "CarbonOS combines hourly dispatch, "
    "balance validation, constraint checking, "
    "cost accounting, carbon accounting and resilience analysis."
)
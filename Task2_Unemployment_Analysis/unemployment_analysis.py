import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Step 1: Load data (two files from the same source)
print("=== Step 1: Load data ===")
df_full = pd.read_csv("data/Unemployment in India.csv")
df_2020 = pd.read_csv("data/Unemployment_Rate_upto_11_2020.csv")
print("File 1 shape:", df_full.shape, "| File 2 shape:", df_2020.shape)

df = df_full.copy()

# Step 2: Data cleaning
#  - strip leading/trailing spaces in column names
#  - convert dates to datetime, extract year/month
print("\n=== Step 2: Data cleaning ===")
df.columns = [c.strip() for c in df.columns]
print("Columns:", df.columns.tolist())
print("\nMissing values before:\n", df.isnull().sum())
df = df.dropna()

def clean(df_in):
    df_in.columns = [c.strip() for c in df_in.columns]
    df_in["Date"] = pd.to_datetime(
        df_in["Date"].astype(str).str.strip(), format="%d-%m-%Y"
    )
    df_in["Year"] = df_in["Date"].dt.year
    df_in["Month"] = df_in["Date"].dt.month
    df_in["MonthYear"] = df_in["Date"].dt.to_period("M")
    for col in [" Estimated Unemployment Rate (%)", " Estimated Employed",
                " Estimated Labour Participation Rate (%)"]:
        if col in df_in.columns:
            df_in[col.strip()] = df_in[col].astype(float)
    df_in = df_in.dropna()
    return df_in

df_full = clean(df_full)
df_2020 = clean(df_2020)
print("Missing values after:",
      df_full["Estimated Unemployment Rate (%)"].isnull().sum())

# Step 3: Overview statistics
print("\n=== Step 3: Overview statistics ===")
print(df_full["Estimated Unemployment Rate (%)"].describe())
print("\nAverage unemployment by Area:\n",
      df_full.groupby("Area")["Estimated Unemployment Rate (%)"].mean())

# Step 4: Visualization - national unemployment over time (file 1, 2019-2020)
print("\n=== Step 4: Visualization ===")
monthly = df_full.groupby("MonthYear")["Estimated Unemployment Rate (%)"].mean()
fig, ax = plt.subplots()
monthly.plot(marker="o", ax=ax)
ax.set_title("Average Unemployment Rate in India (May 2019 - Oct 2020)")
ax.set_xlabel("Month")
ax.set_ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("unemployment_timeline.png", dpi=120)
plt.close()
print("Saved unemployment_timeline.png")

# State-level comparison before vs during Covid (file 2 data covers 2020)
df_2020["Period"] = np.where(df_2020["Date"] < "2020-03-25", "Before Lockdown", "During Lockdown")
state_period = (
    df_2020.groupby(["Region", "Period"])["Estimated Unemployment Rate (%)"]
    .mean()
    .unstack()
)
state_period = state_period.sort_values("During Lockdown", ascending=False)
print("\nTop 10 states by unemployment during lockdown:")
print(state_period.head(10))

ax = state_period.head(10).plot(kind="barh", figsize=(10, 8))
ax.set_title("Unemployment Rate: Before vs During Covid Lockdown (Top 10 States)")
ax.set_xlabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("lockdown_states.png", dpi=120)
plt.close()
print("Saved lockdown_states.png")

# Impact of Covid-19: national averages
print("\n=== Step 5: Covid-19 impact ===")
period_national = df_2020.groupby("Period")["Estimated Unemployment Rate (%)"].mean()
print(period_national.to_string())
delta = (
    period_national.get("During Lockdown", np.nan)
    - period_national.get("Before Lockdown", np.nan)
)
print(f"\nChange in national unemployment rate due to Covid lockdown: {delta:.2f} pp")

# Monthly unemployment trend for the lockdown-impacted months
monthly_2020 = df_2020.groupby("MonthYear")["Estimated Unemployment Rate (%)"].mean()
ax = monthly_2020.plot(marker="o", color="crimson")
ax.set_title("Unemployment Rate in India (Jan - Nov 2020, the Covid-19 period)")
ax.set_xlabel("Month")
ax.set_ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("unemployment_2020.png", dpi=120)
plt.close()
print("Saved unemployment_2020.png")

# Step 6: Seasonal / pattern insights
print("\n=== Step 6: Seasonal patterns ===")
season = df_full.groupby(df_full["Month"])["Estimated Unemployment Rate (%)"].mean()
print(season.to_string())
print("\nSeasonal insight:", round(season.min(), 2),
      "lowest unemployment in month", int(season.idxmin()),
      "|", round(season.max(), 2),
      "highest in month", int(season.idxmax()))

# Step 7: Summary of insights
print("\n=== Step 7: Key insights (for policy) ===")
print("1. Covid-19 produced a massive spike in unemployment during the Q2 2020 lockdown.")
print("2. Rural vs urban split shows which areas need support the most.")
print("3. Recurring seasonal peaks suggest a need for stabilizing policies in those months.")
print("\nRural vs Urban avg (2019-2020):")
print(df_full.groupby("Area")["Estimated Unemployment Rate (%)"].mean().round(2))
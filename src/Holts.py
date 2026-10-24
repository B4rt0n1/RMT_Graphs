import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt
from pathlib import Path

# --- Directory setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_DIR = PROJECT_ROOT / 'csv'
GRAPHS_DIR = PROJECT_ROOT / 'graphs'
CSV_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
filepath = SCRIPT_DIR / 'Лист Microsoft Excel.xlsx'
if not filepath.exists():
    filepath = PROJECT_ROOT / 'Лист Microsoft Excel.xlsx'
xls = pd.ExcelFile(filepath)
ram_df = pd.read_excel(xls, sheet_name='RAM Retail Price')

# 2. Preprocess
ram_df['Date'] = pd.to_datetime(ram_df['Date'])
ram_df.rename(columns={'Price (USD/GB)': 'Price'}, inplace=True)

# Separate DDR4 and DDR5
df_ddr4 = ram_df[ram_df['Type'] == 'DDR4'].sort_values('Date').set_index('Date')
df_ddr5 = ram_df[ram_df['Type'] == 'DDR5'].sort_values('Date').set_index('Date')

# 3. Holt's Forecast Function
def plot_and_save_holt(data, title, color_hist, color_pred, graph_filename, csv_filename, months_to_forecast=16):
    # Resample to monthly frequency using mean and linear interpolation
    ts = data['Price'].resample('MS').mean().interpolate()

    # Fit Holt's Linear Exponential Smoothing model
    model = Holt(ts, initialization_method="estimated").fit()

    # Forecast future values
    forecast = model.forecast(months_to_forecast)
    forecast.name = 'RAM_Price_Forecast'

    # --- Save forecast data to CSV ---
    forecast.to_frame().to_csv(CSV_DIR / csv_filename)
    print(f"[Holt {title}] Forecast saved -> {CSV_DIR / csv_filename}")

    # --- Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(ts.index, ts, marker='o', label='Historical Data', color=color_hist)
    plt.plot(model.fittedvalues.index, model.fittedvalues, linestyle='--', label='Holt Smoothed Fit', color='gray')
    plt.plot(forecast.index, forecast, marker='x', linestyle='-.', label='Forecast (to Jan 2028)', color=color_pred)

    plt.title(f"{title} - Price Forecast using Holt's Linear Smoothing")
    plt.xlabel("Date")
    plt.ylabel("Price (USD/GB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / graph_filename)
    plt.show()
    print(f"[Holt {title}] Graph saved -> {GRAPHS_DIR / graph_filename}")


# 4. Execute and Save Forecasts
plot_and_save_holt(
    df_ddr4, "DDR4", color_hist="blue", color_pred="red",
    graph_filename="holt_ddr4_forecast.png", csv_filename="holt_ddr4_forecast.csv",
    months_to_forecast=16
)
plot_and_save_holt(
    df_ddr5, "DDR5", color_hist="green", color_pred="orange",
    graph_filename="holt_ddr5_forecast.png", csv_filename="holt_ddr5_forecast.csv",
    months_to_forecast=16
)
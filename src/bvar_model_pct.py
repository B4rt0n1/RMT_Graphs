import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
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
ppi_df = pd.read_excel(xls, sheet_name='Semiconductor Price Index')
spend_df = pd.read_excel(xls, sheet_name='Company Spending Data')

# 2. Preprocess Common Indicators
ram_df['Date'] = pd.to_datetime(ram_df['Date'])

ppi_df['Observation Date'] = pd.to_datetime(ppi_df['Observation Date'])
ppi_q = ppi_df.groupby(ppi_df['Observation Date'].dt.to_period('Q'))['Producer Price Index'].mean().reset_index()
ppi_q['Observation Date'] = ppi_q['Observation Date'].dt.to_timestamp()
ppi_q.set_index('Observation Date', inplace=True)
ppi_q.rename(columns={'Producer Price Index': 'PPI'}, inplace=True)

spend_df['Quarter_Clean'] = spend_df['Quarter'].str.extract(r'Q(\d)\s+(\d{4})')[1] + 'Q' + spend_df['Quarter'].str.extract(r'Q(\d)\s+(\d{4})')[0]
spend_df['Quarter_Period'] = pd.PeriodIndex(spend_df['Quarter_Clean'], freq='Q')
spend_df['Date'] = spend_df['Quarter_Period'].dt.to_timestamp()

cost_cols = ['Logic Cost', 'CoWoS Cost', 'HBM Cost', 'Auxiliary Cost']
spend_q = spend_df.groupby('Date')[cost_cols].sum().reset_index()
spend_q.set_index('Date', inplace=True)
spend_q.rename(columns={
    'Logic Cost': 'Logic_Cost',
    'CoWoS Cost': 'CoWoS_Cost',
    'HBM Cost': 'HBM_Cost',
    'Auxiliary Cost': 'Auxiliary_Cost'
}, inplace=True)

# Separate DDR4 and DDR5
ram_ddr4 = ram_df[ram_df['Type'] == 'DDR4'].groupby(ram_df['Date'].dt.to_period('Q'))['Price (USD/GB)'].mean().reset_index()
ram_ddr4['Date'] = ram_ddr4['Date'].dt.to_timestamp()
ram_ddr4.set_index('Date', inplace=True)
ram_ddr4.rename(columns={'Price (USD/GB)': 'RAM_Price'}, inplace=True)

ram_ddr5 = ram_df[ram_df['Type'] == 'DDR5'].groupby(ram_df['Date'].dt.to_period('Q'))['Price (USD/GB)'].mean().reset_index()
ram_ddr5['Date'] = ram_ddr5['Date'].dt.to_timestamp()
ram_ddr5.set_index('Date', inplace=True)
ram_ddr5.rename(columns={'Price (USD/GB)': 'RAM_Price'}, inplace=True)


def run_bvar_forecast(ram_q, title, color_hist, color_pred, graph_filename, csv_filename):
    df_merged = pd.concat([ram_q, ppi_q, spend_q], axis=1).dropna()
    df_merged.index.freq = df_merged.index.inferred_freq

    last_date = df_merged.index[-1]
    target_date = pd.Timestamp('2028-01-01')
    steps = (target_date.year - last_date.year) * 4 + (target_date.quarter - last_date.quarter)
    future_dates = pd.date_range(start=last_date + pd.offsets.QuarterBegin(1), periods=steps, freq='QS-JAN')

    if (df_merged <= 0).any().any():
        raise ValueError("Log differences require all merged series values to be positive.")

    df_log_diff = np.log(df_merged).diff().dropna()

    df_lagged = df_log_diff.shift(1).dropna()
    df_current = df_log_diff.loc[df_lagged.index]
    bvar_model = MultiOutputRegressor(BayesianRidge())
    bvar_model.fit(df_lagged.values, df_current.values)

    current_input = df_log_diff.values[-1].reshape(1, -1)
    predictions_log_diff = []
    for _ in range(steps):
        pred = bvar_model.predict(current_input)
        predictions_log_diff.append(pred[0])
        current_input = pred

    forecast_abs = np.exp(
        np.log(df_merged.iloc[-1].values) + np.cumsum(predictions_log_diff, axis=0)
    )
    bvar_forecast_df = pd.DataFrame(forecast_abs, index=future_dates, columns=df_merged.columns)

    # --- Save forecast data to CSV ---
    bvar_forecast_df.to_csv(CSV_DIR / csv_filename)
    print(f"[BVAR {title}] Forecast saved -> {CSV_DIR / csv_filename}")

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_merged.index, df_merged['RAM_Price'], label=f'Historical {title} Price', color=color_hist, marker='o')
    plt.plot(bvar_forecast_df.index, bvar_forecast_df['RAM_Price'], label='Forecast (log differences) to Jan 2028', color=color_pred, linestyle='--', marker='o')
    plt.title(f'BVAR (Log Differences): {title} Price Projection')
    plt.ylabel('Price (USD/GB)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / graph_filename)
    plt.show()
    print(f"[BVAR {title}] Graph saved -> {GRAPHS_DIR / graph_filename}")


# Run forecast and save plots/CSV for DDR4 and DDR5
run_bvar_forecast(
    ram_ddr4, "DDR4", color_hist="blue", color_pred="red",
    graph_filename="bvar_ddr4_forecast.png", csv_filename="bvar_ddr4_forecast.csv"
)
run_bvar_forecast(
    ram_ddr5, "DDR5", color_hist="green", color_pred="orange",
    graph_filename="bvar_ddr5_forecast.png", csv_filename="bvar_ddr5_forecast.csv"
)
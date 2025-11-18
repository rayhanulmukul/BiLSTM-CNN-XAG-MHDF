# final_demand_forecasting_v13_LSTM_FIX.py
# COMPLETE SYSTEM v13 WITH EDA + CSV CACHING + LSTM TIMELINE FIX + COMPARISON FIGURES
# ✅ FIXED: LSTM timeline generation works with both fresh & cached models
# ✅ NEW: 5 comprehensive comparison figure generation

import warnings, hashlib, argparse
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

import joblib
import tensorflow as tf
from tensorflow.keras import layers, models as keras_models, callbacks, optimizers, regularizers


# ============================================================================
# CUSTOM METRICS & LOSS FUNCTIONS (SAFE MAPE, SMAPE, and Custom Losses)
# ============================================================================
def safe_mape(y_true, y_pred):
    """Calculate MAPE safely with zero handling"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    """Calculate Symmetric MAPE (0-200% bounded)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(numerator[mask] / denominator[mask]) * 100

def mape_loss(y_true, y_pred):
    """MAPE loss for training"""
    epsilon = 1e-10
    diff = tf.abs((y_true - y_pred) / (tf.abs(y_true) + epsilon))
    return tf.reduce_mean(diff) * 100

def smape_loss(y_true, y_pred):
    """SMAPE loss for training"""
    epsilon = 1e-10
    numerator = tf.abs(y_pred - y_true)
    denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2 + epsilon
    return tf.reduce_mean(numerator / denominator) * 100

# ============================================================================
# CONFIGURATION
# ============================================================================
VAL_DAYS   = 365
RS         = 42
FIG_DPI    = 200
LSTM_EPOCHS= 150
BATCH_SIZE = 512
LOOKBACK   = 28
LSTM_UNITS = 128

np.random.seed(RS)
tf.random.set_seed(RS)

ap = argparse.ArgumentParser(description="Complete Demand Forecasting System with CSV Cache")
ap.add_argument("--retrain-lstm", action="store_true", help="Force retrain LSTM")
ap.add_argument("--retrain-ml",   action="store_true", help="Force retrain ML models")
ap.add_argument("--retrain-all",  action="store_true", help="Force retrain all")
ap.add_argument("--skip-eda",     action="store_true", help="Skip EDA (faster)")
ap.add_argument("--force-models", action="store_true", help="Force retrain even if CSV exists")
args = ap.parse_args()
RETRAIN_LSTM = args.retrain_lstm or args.retrain_all
RETRAIN_ML = args.retrain_ml or args.retrain_all
SKIP_EDA = args.skip_eda
FORCE_MODELS = args.force_models

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def plot_timeline(ax, dates, actual, pred, future_dates, future_pred, title):
    """Plot timeline with historical data showing actual and predicted values.
    Each subplot shows a different time window of historical data:
    - 1-month subplot: Shows last 1 month of historical data
    - 3-month subplot: Shows last 3 months of historical data
    - 1-year subplot: Shows last 1 year of historical data
    """
    # Convert to pandas datetime for easier windowing
    pd_dates = pd.to_datetime(dates)
    last_date = pd_dates.max()  # Last historical date
    
    # Determine window size from title
    if "1 Month" in title:
        window_delta = pd.DateOffset(months=1)
    elif "3 Months" in title:
        window_delta = pd.DateOffset(months=3)
    else:  # 1 Year
        window_delta = pd.DateOffset(years=1)
    
    # Set window to show historical data only
    window_start = last_date - window_delta
    historical_mask = (pd_dates >= window_start) & (pd_dates <= last_date)
    
    # Plot historical data only - no future predictions
    ax.plot(pd_dates[historical_mask], actual[historical_mask], 
            label="Actual", linewidth=2.5,
            color='#2c3e50', marker='o', markersize=5, zorder=3)
    ax.plot(pd_dates[historical_mask], pred[historical_mask],
            label="Predicted", linewidth=2.5,
            color='#3498db', alpha=0.85, marker='s', 
            markersize=4, linestyle='--', zorder=2)

    try:
        # Set x-axis limits exactly to the historical data window
            try:
                if len(pd_dates[historical_mask]) > 0:
                    xmin = pd_dates[historical_mask].min()  # Start of window
                    xmax = pd_dates[historical_mask].max()  # End of window
                    ax.set_xlim(xmin, xmax)
                    
                    # Adjust the number of x-ticks based on the window size
                    if "1 Month" in title:  # 1-month
                        ax.xaxis.set_major_locator(plt.MaxNLocator(6))  # Fewer ticks for monthly view
                    elif "3 Months" in title:  # 3-months
                        ax.xaxis.set_major_locator(plt.MaxNLocator(8))
                    else:  # 1-year
                        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
                    
                    # Remove spines and ticks on right and top
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
            except Exception:
                pass
    except Exception:
        # keep plotting resilient
        pass

    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax.set_ylabel("Units Sold", fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')

    # Format dates with appropriate rotation
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    # Ensure there is no extra white space on x-axis beyond our data window
    try:
        if 'xmin' in locals() and 'xmax' in locals():
            ax.set_xlim(xmin, xmax)
            # remove any axis margins
            ax.margins(x=0)
            ax.set_xmargin(0)
            ax.figure.canvas.draw_idle()
    except Exception:
        pass


def plot_metrics_comparison(ax, actual, pred, future_1m=None, future_3m=None, future_1y=None, model_name=None):
    """Render compact metrics summary for a single model on the provided Axes.
    Shows historical metrics (RMSE, MAE, R², MAPE, SMAPE) for three time windows:
    - Last 1 month
    - Last 3 months
    - Last 1 year
    """
    try:
        actual = np.array(actual)
        pred = np.array(pred)

        if len(actual) == 0 or len(pred) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        # Calculate metrics for different time windows
        periods = {
            '1 Month': 30,
            '3 Months': 90,
            '1 Year': 365
        }

        lines = [f"Model: {model_name or 'Model'}", ""]

        for window_name, n_days in periods.items():
            # Take last n days of data
            window_actual = actual[-min(n_days, len(actual)):]
            window_pred = pred[-min(n_days, len(pred)):]
            
            if len(window_actual) > 0 and len(window_pred) > 0:
                rmse = float(np.sqrt(np.mean((window_actual - window_pred) ** 2)))
                mae = float(np.mean(np.abs(window_actual - window_pred)))
                r2 = float(r2_score(window_actual, window_pred)) if len(window_actual) > 1 else float('nan')
                mape_v = safe_mape(window_actual, window_pred)
                smape_v = smape(window_actual, window_pred)

                lines.extend([
                    f"{window_name}:",
                    f"  RMSE : {rmse:.3f}",
                    f"  MAE  : {mae:.3f}",
                    f"  R²   : {r2:.3f}",
                    f"  MAPE : {mape_v:.2f}%",
                    f"  SMAPE: {smape_v:.2f}%",
                    ""
                ])

        txt = "\n".join(lines)
        ax.axis('off')
        ax.text(0.01, 0.99, txt, fontsize=10, va='top', fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1.0))
    except Exception as e:
        ax.axis('off')
        ax.text(0.5, 0.5, f"Error computing metrics:\n{str(e)}", ha='center', va='center')

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def feature_signature(features):
    s = "|".join(features)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def save_fig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def print_section(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def _thin_ticks(ax, labels, num_ticks=12):
    """Helper to thin out x-axis labels"""
    labels = list(labels)
    if not labels: return
    n = len(labels)
    idx = np.linspace(0, n-1, min(num_ticks, n), dtype=int)
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=45, ha="right")

# ============================================================================
# CHECK FOR EXISTING CSV FILES
# ============================================================================

def check_existing_models(OUT_ROOT):
    """Check if model results already exist and load them"""
    ml_csv_path = OUT_ROOT / "01_ml_models_comparison.csv"
    lstm_csv_path = OUT_ROOT / "03_lstm_model_comparison.csv"
    all_csv_path = OUT_ROOT / "02_all_models_comparison.csv"

    if ml_csv_path.exists() and lstm_csv_path.exists() and all_csv_path.exists():
        print("\n ✅ FOUND EXISTING MODEL RESULTS!")
        print(f" • ML Models CSV: {ml_csv_path}")
        print(f" • LSTM CSV: {lstm_csv_path}")
        print(f" • All Models CSV: {all_csv_path}")

        ml_df = pd.read_csv(ml_csv_path)
        lstm_df = pd.read_csv(lstm_csv_path)
        all_df = pd.read_csv(all_csv_path)

        print(f"\n 📊 Loaded Results:")
        print(f" ML Models ({len(ml_df)} rows):")
        print(ml_df.to_string(index=False))
        print(f"\n LSTM Model:")
        print(lstm_df.to_string(index=False))

        return ml_df, lstm_df, all_df, True  # True = loaded from cache

    return None, None, None, False  # False = need to train

# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def perform_eda(train, OUT_ROOT):
    """Comprehensive EDA of the dataset"""
    print_section("PHASE 0: EXPLORATORY DATA ANALYSIS (EDA)")

    EDA_DIR = OUT_ROOT / "eda"
    ensure_dir(EDA_DIR)

    print("\n 📊 Performing comprehensive EDA...")

    # 1. Sales Distribution
    print(" ✓ Generating sales distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(train['sales'], bins=50, color='#4ecdc4', edgecolor='black', alpha=0.7)
    ax.set_title("Sales Distribution", fontweight='bold', fontsize=12)
    ax.set_xlabel("Units Sold")
    ax.set_ylabel("Frequency")
    ax.axvline(train['sales'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {train["sales"].mean():.2f}')
    ax.legend()

    ax = axes[0, 1]
    ax.boxplot(train['sales'], vert=True)
    ax.set_title("Sales Box Plot", fontweight='bold', fontsize=12)
    ax.set_ylabel("Units Sold")
    stats_text = f"Min: {train['sales'].min():.0f}\nMax: {train['sales'].max():.0f}\nMedian: {train['sales'].median():.0f}\nStd: {train['sales'].std():.2f}"
    ax.text(1.15, train['sales'].mean(), stats_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax = axes[1, 0]
    ax.hist(np.log1p(train['sales']), bins=50, color='#ff6b6b', edgecolor='black', alpha=0.7)
    ax.set_title("Log Sales Distribution", fontweight='bold', fontsize=12)
    ax.set_xlabel("Log(Units Sold)")
    ax.set_ylabel("Frequency")

    ax = axes[1, 1]
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    values = [np.percentile(train['sales'], p) for p in percentiles]
    ax.bar([str(p) for p in percentiles], values, color='#95e1d3', edgecolor='black', alpha=0.7)
    ax.set_title("Sales Percentiles", fontweight='bold', fontsize=12)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Units Sold")

    save_fig(EDA_DIR / "01_sales_distribution.png")
    print(" → 01_sales_distribution.png")

    # 2. Temporal Patterns
    print(" ✓ Generating temporal patterns...")
    train['date'] = pd.to_datetime(train['date'])
    train['dow'] = train['date'].dt.dayofweek
    train['month'] = train['date'].dt.month
    train['week'] = train['date'].dt.isocalendar().week

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dow_data = train.groupby('dow')['sales'].agg(['mean', 'std'])
    ax = axes[0, 0]
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.bar(range(7), dow_data['mean'], yerr=dow_data['std'], capsize=5, color='#4ecdc4', alpha=0.7, edgecolor='black')
    ax.set_title("Average Sales by Day of Week", fontweight='bold', fontsize=12)
    ax.set_xlabel("Day")
    ax.set_ylabel("Average Units Sold")
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_labels)

    month_data = train.groupby('month')['sales'].agg(['mean', 'std'])
    ax = axes[0, 1]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.plot(range(1, 13), month_data['mean'], marker='o', linewidth=2, markersize=8, color='#ff6b6b')
    ax.fill_between(range(1, 13),
        month_data['mean'] - month_data['std'],
        month_data['mean'] + month_data['std'],
        alpha=0.3, color='#ff6b6b')
    ax.set_title("Average Sales by Month", fontweight='bold', fontsize=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Units Sold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, rotation=45)
    ax.grid(True, alpha=0.3)

    daily_avg = train.groupby('date')['sales'].mean()
    ax = axes[1, 0]
    ax.plot(daily_avg.index, daily_avg.values, linewidth=1, color='#95e1d3', alpha=0.7)
    ax.set_title("Daily Average Sales Trend", fontweight='bold', fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Units Sold")
    ax.grid(True, alpha=0.3)

    store_sales = train.groupby('store')['sales'].agg(['sum', 'mean', 'count'])
    top_stores = store_sales.nlargest(10, 'sum')
    ax = axes[1, 1]
    ax.barh(range(len(top_stores)), top_stores['sum'], color='#f7b731', edgecolor='black', alpha=0.7)
    ax.set_title("Top 10 Stores by Total Sales", fontweight='bold', fontsize=12)
    ax.set_xlabel("Total Units Sold")
    ax.set_yticks(range(len(top_stores)))
    ax.set_yticklabels([f"Store {i}" for i in top_stores.index])

    save_fig(EDA_DIR / "02_temporal_patterns.png")
    print(" → 02_temporal_patterns.png")

    # 3. Store & Item Analysis
    print(" ✓ Generating store & item analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    store_counts = train['store'].value_counts().sort_index()
    ax = axes[0, 0]
    ax.hist(store_counts.values, bins=30, color='#4ecdc4', edgecolor='black', alpha=0.7)
    ax.set_title(f"Distribution of Records per Store\n(Total: {train['store'].nunique()} stores)", fontweight='bold', fontsize=12)
    ax.set_xlabel("Records per Store")
    ax.set_ylabel("Number of Stores")

    item_counts = train['item'].value_counts().sort_index()
    ax = axes[0, 1]
    ax.hist(item_counts.values, bins=30, color='#ff6b6b', edgecolor='black', alpha=0.7)
    ax.set_title(f"Distribution of Records per Item\n(Total: {train['item'].nunique()} items)", fontweight='bold', fontsize=12)
    ax.set_xlabel("Records per Item")
    ax.set_ylabel("Number of Items")

    store_item_combos = train.groupby(['store', 'item']).size()
    ax = axes[1, 0]
    ax.hist(store_item_combos.values, bins=50, color='#95e1d3', edgecolor='black', alpha=0.7)
    ax.set_title(f"Distribution of Records per Store-Item Combo\n(Total combos: {len(store_item_combos)})", fontweight='bold', fontsize=12)
    ax.set_xlabel("Records per Store-Item")
    ax.set_ylabel("Frequency")

    item_std = train.groupby('item')['sales'].std().nlargest(10)
    ax = axes[1, 1]
    ax.barh(range(len(item_std)), item_std.values, color='#f7b731', edgecolor='black', alpha=0.7)
    ax.set_title("Top 10 Items by Sales Volatility (Std Dev)", fontweight='bold', fontsize=12)
    ax.set_xlabel("Standard Deviation of Sales")
    ax.set_yticks(range(len(item_std)))
    ax.set_yticklabels([f"Item {i}" for i in item_std.index])

    save_fig(EDA_DIR / "03_store_item_analysis.png")
    print(" → 03_store_item_analysis.png")

    # 4. Autocorrelation Analysis
    print(" ✓ Generating autocorrelation analysis...")
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    samples = train.groupby(['store', 'item']).size().nlargest(3).index.tolist()

    for idx, (store, item) in enumerate(samples[:2]):
        data = train[(train['store'] == store) & (train['item'] == item)].sort_values('date')['sales'].values
        ax = axes[0, idx]
        plot_acf(data, lags=30, ax=ax, title=f'ACF - Store {store}, Item {item}')
        ax.set_xlabel("Lag")

        ax = axes[1, idx]
        plot_pacf(data, lags=30, ax=ax, title=f'PACF - Store {store}, Item {item}')
        ax.set_xlabel("Lag")

    all_sales = train.sort_values('date')['sales'].values
    if len(samples) < 2:
        ax = axes[0, 1]
        plot_acf(all_sales, lags=30, ax=ax, title='ACF - All Sales')
        ax.set_xlabel("Lag")

        ax = axes[1, 1]
        plot_pacf(all_sales, lags=30, ax=ax, title='PACF - All Sales')
        ax.set_xlabel("Lag")

    save_fig(EDA_DIR / "04_autocorrelation_analysis.png")
    print(" → 04_autocorrelation_analysis.png")

    # 5. Statistical Summary
    print(" ✓ Generating statistical summary...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    stats_summary = f"""
╔═══════════════════════════════════════════════════════════════╗
║ DATASET STATISTICAL SUMMARY ║
╚═══════════════════════════════════════════════════════════════╝

📊 OVERALL DATASET:
• Total Records: {len(train):,}
• Date Range: {train['date'].min().date()} to {train['date'].max().date()}
• Duration: {(train['date'].max() - train['date'].min()).days} days

🏪 STORES & ITEMS:
• Number of Stores: {train['store'].nunique()}
• Number of Items: {train['item'].nunique()}
• Store-Item Combinations: {train.groupby(['store', 'item']).ngroups}
• Avg Records per Store: {len(train) / train['store'].nunique():.0f}
• Avg Records per Item: {len(train) / train['item'].nunique():.0f}

📈 SALES STATISTICS:
• Mean Sales: {train['sales'].mean():.2f} units
• Median Sales: {train['sales'].median():.2f} units
• Std Dev: {train['sales'].std():.2f} units
• Min Sales: {train['sales'].min():.0f} units
• Max Sales: {train['sales'].max():.0f} units
• Skewness: {train['sales'].skew():.4f}
• Kurtosis: {train['sales'].kurtosis():.4f}

🔍 MISSING VALUES:
• Missing in 'sales': {train['sales'].isna().sum()}
• Missing in 'date': {train['date'].isna().sum()}
• Missing in 'store': {train['store'].isna().sum()}
• Missing in 'item': {train['item'].isna().sum()}

📅 TEMPORAL INSIGHTS:
• Most Active Day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][train.groupby('dow')['sales'].mean().idxmax()]}
• Least Active Day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][train.groupby('dow')['sales'].mean().idxmin()]}
• Peak Month: {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][train.groupby('month')['sales'].mean().idxmax()-1]}
• Low Month: {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][train.groupby('month')['sales'].mean().idxmin()-1]}

🎯 TOP PERFORMERS:
• Top Store: Store {train.groupby('store')['sales'].sum().idxmax()} (Total: {train.groupby('store')['sales'].sum().max():,.0f} units)
• Top Item: Item {train.groupby('item')['sales'].sum().idxmax()} (Total: {train.groupby('item')['sales'].sum().max():,.0f} units)
• Most Volatile Item: Item {train.groupby('item')['sales'].std().idxmax()} (Std: {train.groupby('item')['sales'].std().max():.2f})
"""

    ax.text(0.05, 0.95, stats_summary, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_fig(EDA_DIR / "05_statistical_summary.png")
    print(" → 05_statistical_summary.png")

    print("\n ✅ EDA completed!")
    return EDA_DIR

# ============================================================================
# IMPROVED HYBRID LSTM WITH ATTENTION MECHANISM
# ============================================================================

@tf.keras.utils.register_keras_serializable(package='custom_layers')
class AttentionSqueeze(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        return tf.squeeze(x, axis=-1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable(package='custom_layers')
class AttentionSum(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, x):
        return tf.reduce_sum(x, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def get_config(self):
        config = super().get_config()
        return config

def build_improved_hybrid_lstm_with_attention(lookback, n_tabular_features):
    """🚀 PREMIUM HYBRID LSTM ARCHITECTURE WITH ATTENTION"""

    seq_input = layers.Input(shape=(lookback, 1), name='sequential_input')

    x_lstm = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True,
        kernel_regularizer=regularizers.l2(1e-5))
    )(seq_input)
    x_lstm = layers.BatchNormalization()(x_lstm)
    x_lstm = layers.Dropout(0.3)(x_lstm)

    attention = layers.Dense(1, activation='sigmoid')(x_lstm)
    attention = AttentionSqueeze()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(LSTM_UNITS * 2)(attention)
    attention = layers.Permute([2, 1])(attention)

    x_lstm_att = layers.Multiply()([x_lstm, attention])
    x_lstm_att = AttentionSum()(x_lstm_att)

    x_lstm_2 = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS // 2, return_sequences=False,
        kernel_regularizer=regularizers.l2(1e-5))
    )(x_lstm)
    x_lstm_2 = layers.BatchNormalization()(x_lstm_2)
    x_lstm_2 = layers.Dropout(0.3)(x_lstm_2)

    x_lstm_out = layers.Concatenate()([x_lstm_att, x_lstm_2])
    x_lstm_out = layers.Dense(64, activation='relu')(x_lstm_out)
    x_lstm_out = layers.Dropout(0.2)(x_lstm_out)

    tabular_input = layers.Input(shape=(n_tabular_features,), name='tabular_input')

    x_tab = layers.Dense(256, activation=None,
        kernel_regularizer=regularizers.l2(1e-5))(tabular_input)
    x_tab = layers.BatchNormalization()(x_tab)
    x_tab = layers.Activation('relu')(x_tab)
    x_tab = layers.Dropout(0.3)(x_tab)

    x_tab_res1 = layers.Dense(256, activation=None,
        kernel_regularizer=regularizers.l2(1e-5))(x_tab)
    x_tab_res1 = layers.BatchNormalization()(x_tab_res1)
    x_tab_res1 = layers.Add()([x_tab, x_tab_res1])
    x_tab_res1 = layers.Activation('relu')(x_tab_res1)
    x_tab_res1 = layers.Dropout(0.3)(x_tab_res1)

    x_tab_res2 = layers.Dense(128, activation=None,
        kernel_regularizer=regularizers.l2(1e-5))(x_tab_res1)
    x_tab_res2 = layers.BatchNormalization()(x_tab_res2)
    x_tab_proj = layers.Dense(128)(x_tab_res1)

    x_tab = layers.Add()([x_tab_res2, x_tab_proj])
    x_tab = layers.Activation('relu')(x_tab)
    x_tab = layers.Dropout(0.3)(x_tab)

    x_tab = layers.Dense(64, activation='relu')(x_tab)
    x_tab = layers.Dropout(0.2)(x_tab)

    combined = layers.Concatenate()([x_lstm_out, x_tab])

    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation='relu')(x)

    out = layers.Dense(1, activation=None)(x)

    model = keras_models.Model(inputs=[seq_input, tabular_input], outputs=out)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss='huber',
        metrics=['mae']
    )

    return model

# ============================================================================
# DATA PREPARATION FOR HYBRID LSTM
# ============================================================================

def create_hybrid_lstm_data(work, cutoff, features, lookback=28):
    """Prepare dual-input data for hybrid LSTM"""
    print(f"\n 📊 Creating hybrid LSTM sequences (lookback={lookback})...")

    X_train_seq, X_train_tab, y_train = [], [], []
    X_val_seq, X_val_tab, y_val = [], [], []

    total = work.groupby(['store','item']).ngroups

    for idx, ((store, item), group) in enumerate(work.groupby(['store','item'])):
        group = group.sort_values('date').reset_index(drop=True)
        train_g = group[group['date'] <= cutoff]
        val_g = group[group['date'] > cutoff]

        if len(train_g) > lookback:
            for i in range(lookback, len(train_g)):
                seq = train_g['sales'].iloc[i-lookback:i].values.reshape(-1, 1)
                tab = train_g[features].iloc[i].values
                X_train_seq.append(seq)
                X_train_tab.append(tab)
                y_train.append(train_g['sales'].iloc[i])

        if len(val_g) > lookback:
            for i in range(lookback, len(val_g)):
                seq = val_g['sales'].iloc[i-lookback:i].values.reshape(-1, 1)
                tab = val_g[features].iloc[i].values
                X_val_seq.append(seq)
                X_val_tab.append(tab)
                y_val.append(val_g['sales'].iloc[i])

        if (idx + 1) % 200 == 0:
            print(f" ✓ Processed {idx+1}/{total} store-item groups...")

    data = {
        'X_train_seq': np.array(X_train_seq),
        'X_train_tab': np.array(X_train_tab),
        'y_train': np.array(y_train),
        'X_val_seq': np.array(X_val_seq),
        'X_val_tab': np.array(X_val_tab),
        'y_val': np.array(y_val)
    }

    print(f" ✓ Train: seq{data['X_train_seq'].shape} tab{data['X_train_tab'].shape}")
    print(f" ✓ Val: seq{data['X_val_seq'].shape} tab{data['X_val_tab'].shape}")

    return data

# ============================================================================
# ✅ FIXED TIMELINE GENERATION - SEPARATE ML & LSTM
# ============================================================================

def generate_ml_timelines(work, cutoff, y_val, ml_predictions, ml_models, X_val, FIG_DIR):
    """Generate timelines for 6 ML models with future predictions.

    Notes:
    - ml_predictions: dict of model_name -> predictions for X_val
    - ml_models: dict of model_name -> fitted model (not required for plotting but kept
      for parity with the rest of the code)
    - X_val: the validation features (passed so callers don't rely on globals)
    """
    print("\n 📈 Generating timeline plots for 6 ML models with future predictions...")

    val_indices = work["date"] > cutoff
    val_dates = work.loc[val_indices, "date"].values
    val_actual = work.loc[val_indices, "sales"].values
    val_store = work.loc[val_indices, "store"].values
    val_item = work.loc[val_indices, "item"].values

    # Generate future dates for predictions (exactly matching the horizon)
    last_date = work["date"].max()
    forecast_start = last_date + pd.Timedelta(days=1)
    future_dates_1m = pd.date_range(start=forecast_start, periods=30, freq='D')  # 1 month
    future_dates_3m = pd.date_range(start=forecast_start, periods=90, freq='D')  # 3 months
    future_dates_1y = pd.date_range(start=forecast_start, periods=365, freq='D') # 1 year

    print(f" ✓ Validation samples: {len(val_actual)}")

    # Find a common store-item pair (try defaults then fall back to most common)
    store_id, item_id = 1, 1
    store_mask_ml = val_store == store_id
    item_mask_ml = val_item == item_id
    pair_mask_ml = store_mask_ml & item_mask_ml

    if not pair_mask_ml.any():
        pair_mask_ml = (val_store == 0) & (val_item == 0)

    if not pair_mask_ml.any():
        most_common = pd.Series(list(zip(val_store, val_item))).value_counts().index[0]
        store_id, item_id = most_common
        pair_mask_ml = (val_store == store_id) & (val_item == item_id)

    model_names = list(ml_predictions.keys())
    colors = ['#4ecdc4', '#ff6b6b', '#95e1d3', '#f7b731', '#5f27cd', '#00d2d3']

    for idx, model_name in enumerate(model_names):
        try:
            # Historical predictions
            pred_values = np.array(ml_predictions[model_name])[pair_mask_ml]
            actual_values = np.array(val_actual)[pair_mask_ml]
            dates = np.array(val_dates)[pair_mask_ml]

            if len(pred_values) == 0 or len(actual_values) == 0:
                print(f" ⚠️ No validation data for {model_name} on store {store_id}, item {item_id}")
                continue

            # Simple future placeholder: repeat last prediction for each future day
            future_pred_1m = np.full(30, pred_values[-1]) if len(pred_values) > 0 else np.full(30, np.nan)
            future_pred_3m = np.full(90, pred_values[-1]) if len(pred_values) > 0 else np.full(90, np.nan)
            future_pred_1y = np.full(365, pred_values[-1]) if len(pred_values) > 0 else np.full(365, np.nan)

            # Create subplots with less height and tighter spacing
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 16))
            plt.subplots_adjust(hspace=0.4)  # Adjust space between subplots

            # Historical + futures visualized per horizon
            plot_timeline(ax1, dates[-30:], actual_values[-30:], pred_values[-30:], 
                        future_dates_1m, future_pred_1m,
                        title=f"{model_name} - 1 Month Ahead (Store {store_id}, Item {item_id})")
            
            plot_timeline(ax2, dates[-90:], actual_values[-90:], pred_values[-90:], 
                        future_dates_3m, future_pred_3m,
                        title=f"{model_name} - 3 Months Ahead (Store {store_id}, Item {item_id})")
            
            plot_timeline(ax3, dates[-365:], actual_values[-365:], pred_values[-365:], 
                        future_dates_1y, future_pred_1y,
                        title=f"{model_name} - 1 Year Ahead (Store {store_id}, Item {item_id})")

            # Metrics comparison (pass future arrays as available)
            plot_metrics_comparison(ax4, actual_values, pred_values,
                                    future_1m=future_pred_1m, future_3m=future_pred_3m, future_1y=future_pred_1y,
                                    model_name=model_name)

            # Save
            timeline_dir = FIG_DIR / "timelines"
            ensure_dir(timeline_dir)
            filename = f"timeline_{model_name}_store_{store_id}__item_{item_id}.png"
            save_fig(timeline_dir / filename)
            print(f" ✓ {filename}")

        except Exception as e:
            print(f" ❌ Error generating {model_name}: {str(e)}\n")
            plt.close()
            continue
        

# ============================================================================
# ✅ NEW: COMPARISON FIGURES GENERATION
# ============================================================================

def generate_lstm_timeline(work, cutoff, lstm_data, y_lstm_pred, lstm_model, tab_scaler, lstm_scaler, FIG_DIR):
    """Generate timeline for LSTM model with future predictions.

    lstm_model, tab_scaler and lstm_scaler must be provided (they are
    created/loaded in main()).
    """
    print("\n 📈 Generating timeline plot for ImprovedHybridLSTM with future predictions...")

    try:
        # Create validation mask for LSTM data
        val_indices = work["date"] > cutoff
        val_dates = work.loc[val_indices, "date"].values
        val_actual = work.loc[val_indices, "sales"].values
        val_store = work.loc[val_indices, "store"].values
        val_item = work.loc[val_indices, "item"].values

        # Use LSTM validation data directly (already has correct length)
        lstm_val_dates = val_dates[:len(lstm_data['y_val'])]
        lstm_val_store = val_store[:len(lstm_data['y_val'])]
        lstm_val_item = val_item[:len(lstm_data['y_val'])]

        # Find store-item pair
        store_id, item_id = 1, 1
        store_mask = lstm_val_store == store_id
        item_mask = lstm_val_item == item_id
        pair_mask = store_mask & item_mask

        if not pair_mask.any():
            most_common = pd.Series(list(zip(lstm_val_store, lstm_val_item))).value_counts().index[0]
            store_id, item_id = most_common
            pair_mask = (lstm_val_store == store_id) & (lstm_val_item == item_id)

        print(f" ✓ LSTM Timeline: store={store_id}, item={item_id}")
        print(f" ✓ LSTM samples used: {pair_mask.sum()}")

        # Extract data
        pred_values = np.array(y_lstm_pred)[pair_mask]
        actual_values = np.array(lstm_data['y_val'])[pair_mask]
        dates = np.array(lstm_val_dates)[pair_mask]

        if len(pred_values) == 0 or len(actual_values) == 0:
            print(" ⚠️ No data found for this store-item pair!")
            return

        # Generate future dates
        last_date = work["date"].max()
        future_dates_1m = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
        future_dates_3m = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90, freq='D')
        future_dates_1y = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365, freq='D')

        # Generate future predictions using LSTM recursive helper
        # Use the last validation sequence for the selected store-item pair so the
        # future predictions correspond to that pair (not a global last sample).
        pair_indices = np.where(pair_mask)[0]
        if len(pair_indices) > 0:
            last_idx = pair_indices[-1]
            last_seq = lstm_data['X_val_seq'][last_idx:last_idx+1]
            last_tab = lstm_data['X_val_tab'][last_idx:last_idx+1]
        else:
            # fallback to global last sample
            last_seq = lstm_data['X_val_seq'][-1:]
            last_tab = lstm_data['X_val_tab'][-1:]

        future_pred_1m = generate_future_predictions_lstm(lstm_model, last_seq, last_tab, 30, lstm_scaler, tab_scaler)
        future_pred_3m = generate_future_predictions_lstm(lstm_model, last_seq, last_tab, 90, lstm_scaler, tab_scaler)
        future_pred_1y = generate_future_predictions_lstm(lstm_model, last_seq, last_tab, 365, lstm_scaler, tab_scaler)

        # Create subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 20))

        # Historical + 1 Month
        plot_timeline(ax1, dates, actual_values, pred_values, future_dates_1m, future_pred_1m,
                      title=f"ImprovedHybridLSTM - 1 Month Ahead (Store {store_id}, Item {item_id})")

        # Historical + 3 Months
        plot_timeline(ax2, dates, actual_values, pred_values, future_dates_3m, future_pred_3m,
                      title=f"ImprovedHybridLSTM - 3 Months Ahead (Store {store_id}, Item {item_id})")

        # Historical + 1 Year
        plot_timeline(ax3, dates, actual_values, pred_values, future_dates_1y, future_pred_1y,
                      title=f"ImprovedHybridLSTM - 1 Year Ahead (Store {store_id}, Item {item_id})")

        # Metrics Comparison
        plot_metrics_comparison(ax4, actual_values, pred_values,
                                future_1m=future_pred_1m, future_3m=future_pred_3m, future_1y=future_pred_1y,
                                model_name="ImprovedHybridLSTM")

        # Save
        timeline_dir = FIG_DIR / "timelines"
        ensure_dir(timeline_dir)
        filename = f"timeline_ImprovedHybridLSTM_store_{store_id}__item_{item_id}.png"
        save_fig(timeline_dir / filename)
        print(f" ✅ {filename} - SAVED SUCCESSFULLY")
        print(f" - Actual: mean={actual_values.mean():.2f}, std={actual_values.std():.2f}")
        print(f" - Pred: mean={pred_values.mean():.2f}, std={pred_values.std():.2f}")
        mae = np.mean(np.abs(actual_values - pred_values))
        rmse = np.sqrt(np.mean((actual_values - pred_values)**2))
        print(f" - MAE: {mae:.2f}, RMSE: {rmse:.2f}\n")

    except Exception as e:
        print(f" ❌ Error generating LSTM timeline: {str(e)}\n")
        import traceback
        traceback.print_exc()
        plt.close()

def generate_future_predictions(model, data, n_steps, scaler=None):
    """Generate future predictions recursively"""
    predictions = []
    current_data = data.copy()
    
    for _ in range(n_steps):
        pred = model.predict(current_data)
        if scaler is not None:
            pred = scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
        predictions.append(pred[0])
        
        # Update features for next prediction
        current_data = np.roll(current_data, -1)
        current_data[0, -1] = pred[0]
    
    return np.array(predictions)


def generate_future_predictions_lstm(model, last_seq, last_tab, n_steps, lstm_scaler, tab_scaler=None):
    """Generate future predictions for the hybrid LSTM by recursive forecasting.

    - last_seq: ndarray shape (1, lookback, 1)
    - last_tab: ndarray shape (1, n_tab_features)
    Returns a 1D array with n_steps values (in original scale).
    """
    preds = []

    seq = last_seq.copy()
    tab = last_tab.copy()
    if tab_scaler is not None:
        tab_z = tab_scaler.transform(tab)
    else:
        tab_z = tab

    for _ in range(n_steps):
        pred_z = model.predict([seq, tab_z], verbose=0).ravel()[0]
        # inverse
        try:
            pred = lstm_scaler.inverse_transform(np.array(pred_z).reshape(-1, 1)).ravel()[0]
        except Exception:
            pred = float(pred_z)

        preds.append(pred)

        # roll sequence and append predicted value
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, 0] = pred

        # NOTE: tabular features are kept constant; for more realistic forecasts
        # you'd update time features (date/month/dow) and rolling stats.

    return np.array(preds)


def generate_comparison_figures(ml_df, lstm_df, all_df, FIG_DIR):
    """Generate comprehensive comparison figures including future predictions"""

    print_section("PHASE 5: GENERATING COMPARISON FIGURES")

    COMP_DIR = FIG_DIR / "comparisons"
    ensure_dir(COMP_DIR)

    # ===== FIGURE 1: RMSE Comparison =====
    print(" ✓ Generating 01_rmse_comparison.png...")
    fig, ax = plt.subplots(figsize=(14, 7))

    models = all_df['Model'].values
    rmse_vals = all_df['Validation RMSE'].values if 'Validation RMSE' in all_df.columns else all_df['RMSE'].values

    colors = ['#17a2b8' if m != 'CustomLSTM' else '#dc3545' for m in models]
    bars = ax.bar(models, rmse_vals, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, rmse_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Highlight best model
    best_idx = np.argmin(rmse_vals)
    bars[best_idx].set_linewidth(3)
    bars[best_idx].set_edgecolor('gold')

    ax.set_title('Model Comparison: Validation RMSE (Lower is Better) ↓', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    plt.xticks(rotation=45, ha='right')

    save_fig(COMP_DIR / "01_rmse_comparison.png")

    # ===== FIGURE 2: MAE Comparison =====
    print(" ✓ Generating 02_mae_comparison.png...")
    fig, ax = plt.subplots(figsize=(14, 7))

    mae_vals = all_df['Validation MAE'].values if 'Validation MAE' in all_df.columns else all_df['MAE'].values

    colors = ['#17a2b8' if m != 'CustomLSTM' else '#dc3545' for m in models]
    bars = ax.bar(models, mae_vals, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)

    for bar, val in zip(bars, mae_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    best_idx = np.argmin(mae_vals)
    bars[best_idx].set_linewidth(3)
    bars[best_idx].set_edgecolor('gold')

    ax.set_title('Model Comparison: Mean Absolute Error (Lower is Better) ↓', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    plt.xticks(rotation=45, ha='right')

    save_fig(COMP_DIR / "02_mae_comparison.png")

    # ===== FIGURE 3: R² Comparison =====
    print(" ✓ Generating 03_r2_comparison.png...")
    fig, ax = plt.subplots(figsize=(14, 7))

    r2_vals = all_df['Validation R²'].values if 'Validation R²' in all_df.columns else all_df['R2'].values

    colors = ['#17a2b8' if m != 'CustomLSTM' else '#dc3545' for m in models]
    bars = ax.bar(models, r2_vals, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)

    for bar, val in zip(bars, r2_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    best_idx = np.argmax(r2_vals)
    bars[best_idx].set_linewidth(3)
    bars[best_idx].set_edgecolor('gold')

    ax.set_title('Model Comparison: R² Score (Higher is Better) ↑', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_ylim([min(r2_vals) * 0.95, 1.0])
    plt.xticks(rotation=45, ha='right')

    save_fig(COMP_DIR / "03_r2_comparison.png")

    # ===== FIGURE 4: MAPE Comparison =====
    print(" ✓ Generating 04_mape_comparison.png...")
    fig, ax = plt.subplots(figsize=(14, 7))

    mape_vals = all_df['Validation MAPE %'].values if 'Validation MAPE %' in all_df.columns else all_df['MAPE'].values

    colors = ['#17a2b8' if m != 'CustomLSTM' else '#dc3545' for m in models]
    bars = ax.bar(models, mape_vals, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)

    for bar, val in zip(bars, mape_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    best_idx = np.argmin(mape_vals)
    bars[best_idx].set_linewidth(3)
    bars[best_idx].set_edgecolor('gold')

    ax.set_title('Model Comparison: MAPE % (Lower is Better) ↓', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('MAPE %', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    plt.xticks(rotation=45, ha='right')

    save_fig(COMP_DIR / "04_mape_comparison.png")

    # ===== FIGURE 5: Multi-Metric Comparison =====
    print(" ✓ Generating 05_multi_metric_comparison.png...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: RMSE
    ax = axes[0, 0]
    colors = ['#17a2b8' if m != 'CustomLSTM' else '#dc3545' for m in models]
    bars = ax.bar(models, rmse_vals, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    for bar, val in zip(bars, rmse_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    best_idx = np.argmin(rmse_vals)
    bars[best_idx].set_linewidth(2.5)
    bars[best_idx].set_edgecolor('gold')
    ax.set_title('RMSE (Lower Better) ↓', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    # Subplot 2: MAE
    ax = axes[0, 1]
    colors = ['#17a2b8' if m != 'CustomLSTM' else '#dc3545' for m in models]
    bars = ax.bar(models, mae_vals, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    for bar, val in zip(bars, mae_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    best_idx = np.argmin(mae_vals)
    bars[best_idx].set_linewidth(2.5)
    bars[best_idx].set_edgecolor('gold')
    ax.set_title('MAE (Lower Better) ↓', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    # Subplot 3: R²
    ax = axes[1, 0]
    colors = ['#17a2b8' if m != 'CustomLSTM' else '#dc3545' for m in models]
    bars = ax.bar(models, r2_vals, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    for bar, val in zip(bars, r2_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    best_idx = np.argmax(r2_vals)
    bars[best_idx].set_linewidth(2.5)
    bars[best_idx].set_edgecolor('gold')
    ax.set_title('R² (Higher Better) ↑', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=10, fontweight='bold')
    ax.set_ylim([min(r2_vals) * 0.95, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    # Subplot 4: MAPE
    ax = axes[1, 1]
    colors = ['#17a2b8' if m != 'CustomLSTM' else '#dc3545' for m in models]
    bars = ax.bar(models, mape_vals, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
    for bar, val in zip(bars, mape_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    best_idx = np.argmin(mape_vals)
    bars[best_idx].set_linewidth(2.5)
    bars[best_idx].set_edgecolor('gold')
    ax.set_title('MAPE % (Lower Better) ↓', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAPE %', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    plt.suptitle('Model Performance: Comprehensive Metrics Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)

    save_fig(COMP_DIR / "05_multi_metric_comparison.png")

    print("\n ✅ All 5 comparison figures generated successfully!")

# ============================================================================
# TREND ANALYSIS AND FORECASTING
# ============================================================================

def analyze_and_forecast_trends(dates, values, future_periods=365):
    """Analyze trends and generate forecasts using multiple methods"""
    from scipy import stats
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    # Linear trend
    x = np.arange(len(dates))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
    linear_trend = slope * np.arange(len(dates) + future_periods) + intercept
    
    # Exponential smoothing
    model = ExponentialSmoothing(
        values,
        seasonal_periods=7,
        trend='add',
        seasonal='add'
    ).fit()
    
    hw_forecast = model.forecast(future_periods)
    
    return {
        'linear_trend': linear_trend,
        'exp_smoothing': np.concatenate([model.fittedvalues, hw_forecast]),
        'slope': slope,
        'r_squared': r_value**2
    }


def plot_metric_comparison(models, values, metric_name, save_path):
    """Simple bar plot generator used by cached-run metric summaries."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#17a2b8' if m != 'CustomLSTM' else '#dc3545' for m in models]
    bars = ax.bar(models, values, color=colors, edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.3f}', ha='center', va='bottom')
    ax.set_title(f'Model Comparison: {metric_name}')
    ax.set_ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    ensure_dir(Path(save_path).parent)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print_section("COMPLETE DEMAND FORECASTING SYSTEM v15 WITH ENHANCED PREDICTIONS")
    print(" EDA + CSV Caching + 6 ML Models + Improved Hybrid LSTM + Extended Timelines + Enhanced Comparisons")

    # ---------- SETUP ----------
    print_section("PHASE 0: SETUP & CSV CACHE CHECK")

    train = pd.read_csv("train.csv")
    train["date"] = pd.to_datetime(train["date"], errors="coerce")

    print(f"\n 📈 Dataset: {len(train):,} rows")

    FEAT_SIG = hashlib.md5("features".encode()).hexdigest()[:10]
    OUT_ROOT = Path("out_final_v13_lstm_fix") / f"feat_{FEAT_SIG}"
    FIG_DIR = OUT_ROOT / "figs"
    LOG_DIR = OUT_ROOT / "logs"

    ensure_dir(OUT_ROOT); ensure_dir(FIG_DIR); ensure_dir(LOG_DIR)

    # ✅ CHECK FOR EXISTING CSVs
    print("\n 🔍 Checking for existing model results...")
    ml_df, lstm_df, all_df, from_cache = check_existing_models(OUT_ROOT)

    # Feature Engineering (ALWAYS DO THIS - needed for timelines)
    work = train.copy().sort_values(["store","item","date"]).reset_index(drop=True)

    work["year"] = work["date"].dt.year.astype("int16")
    work["month"] = work["date"].dt.month.astype("int8")
    work["day"] = work["date"].dt.day.astype("int8")
    work["dow"] = work["date"].dt.dayofweek.astype("int8")
    work["week"] = work["date"].dt.isocalendar().week.astype("int16")
    work["quarter"] = work["date"].dt.quarter.astype("int8")
    work["dayofyear"] = work["date"].dt.dayofyear.astype("int16")
    work["is_weekend"] = work["dow"].isin([5,6]).astype("int8")
    work["is_month_start"] = work["date"].dt.is_month_start.astype("int8")
    work["is_month_end"] = work["date"].dt.is_month_end.astype("int8")
    work["is_year_start"] = work["date"].dt.is_year_start.astype("int8")
    work["is_year_end"] = work["date"].dt.is_year_end.astype("int8")

    g = work.groupby(["store","item"], observed=True)
    for L in [1,7,14,28,60]:
        work[f"lag_{L}"] = g["sales"].shift(L)

    for W in [7,14,28,60]:
        work[f"roll_mean_{W}"] = g["sales"].transform(lambda s: s.shift(1).rolling(W, 1).mean())
        work[f"roll_std_{W}"] = g["sales"].transform(lambda s: s.shift(1).rolling(W, 1).std())

    work = work.dropna(subset=["lag_1","lag_7","lag_14","lag_28","lag_60"]).copy()

    work["store"] = work["store"].astype("category").cat.codes.astype("int32")
    work["item"] = work["item"].astype("category").cat.codes.astype("int32")

    FEATURES = [
        "store","item",
        "year","month","day","dow","week","quarter","dayofyear",
        "is_weekend","is_month_start","is_month_end","is_year_start","is_year_end",
        "lag_1","lag_7","lag_14","lag_28","lag_60",
        "roll_mean_7","roll_mean_14","roll_mean_28","roll_mean_60",
        "roll_std_7","roll_std_14","roll_std_28","roll_std_60",
    ]

    TARGET = "sales"
    cutoff = work["date"].max() - pd.Timedelta(days=VAL_DAYS)

    X_train = work.loc[work["date"] <= cutoff, FEATURES]
    y_train = work.loc[work["date"] <= cutoff, TARGET]
    X_val = work.loc[work["date"] > cutoff, FEATURES]
    y_val = work.loc[work["date"] > cutoff, TARGET]

    print(f"\n ✓ Train: {len(X_train):,} | Val: {len(X_val):,}")

    if from_cache and not FORCE_MODELS:
        print("\n ✅ USING CACHED RESULTS - Skipping model training!")
        print(" ⚠️ Regenerating timelines and comparisons from cached models...\n")

        # Load cached models for timelines
        ml_predictions = {}
        ml_models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=RS),
            "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=2, random_state=RS),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=300, max_depth=12, n_jobs=2, random_state=RS),
            "GradientBoosting": GradientBoostingRegressor(learning_rate=0.05, n_estimators=600, max_depth=3, subsample=0.9, random_state=RS),
            "HistGradientBoosting": HistGradientBoostingRegressor(max_depth=12, learning_rate=0.05, max_iter=600, l2_regularization=1.0, random_state=RS),
        }

        for name in ml_models.keys():
            model_path = OUT_ROOT / f"{name}.joblib"
            if model_path.exists():
                model = joblib.load(model_path)
                ml_predictions[name] = model.predict(X_val)
                # replace placeholder estimator with the loaded, fitted one so later code can use it
                ml_models[name] = model
                print(f" ✓ Loaded {name}")

        # Load LSTM model (if possible). If loading fails we continue without LSTM timelines.
        lstm_path = OUT_ROOT / "ImprovedHybridLSTMv13.keras"
        lstm_model = None
        lstm_data = None
        lstm_scaler = None
        tab_scaler = None
        y_lstm_pred = None

        if lstm_path.exists():
            try:
                lstm_model = keras_models.load_model(
                    lstm_path,
                    safe_mode=False,
                    custom_objects={
                        'AttentionSqueeze': AttentionSqueeze,
                        'AttentionSum': AttentionSum
                    }
                )

                # Recreate LSTM data
                lstm_data = create_hybrid_lstm_data(work, cutoff, FEATURES, lookback=LOOKBACK)

                # Load scalers
                lstm_scaler = joblib.load(OUT_ROOT / "lstm_scaler.joblib")
                tab_scaler = joblib.load(OUT_ROOT / "tab_scaler.joblib")

                # Predict
                y_val_z = lstm_scaler.transform(lstm_data['y_val'].reshape(-1,1)).ravel()
                X_val_tab_z = tab_scaler.transform(lstm_data['X_val_tab'])
                y_lstm_pred_z = lstm_model.predict([lstm_data['X_val_seq'], X_val_tab_z], verbose=0).ravel()
                y_lstm_pred = lstm_scaler.inverse_transform(y_lstm_pred_z.reshape(-1,1)).ravel()

            except Exception as e:
                print(f"⚠️ Could not load cached LSTM model ({lstm_path}): {e}")
                import traceback
                traceback.print_exc()
                print("⚠️ Skipping LSTM timelines — run with --retrain-all to retrain or fix model file compatibility.")

    # Generate visualizations and comparisons
        print_section("PHASE 4: GENERATING VISUALIZATIONS")

    # 1. Generate ML model timelines
        print("\n🔄 Generating ML model timelines...")
        # Use the consolidated generator which handles cached fitted models correctly
        try:
            generate_ml_timelines(work, cutoff, y_val, ml_predictions, ml_models, X_val, FIG_DIR)
        except Exception as e:
            print(f"⚠️ Error generating ML timelines: {e}")

        # 2. Generate comparison figures
        try:
            print("\n🔄 Generating comparison figures...")
            # use the new consolidated generator
            generate_comparison_figures(ml_df, lstm_df, all_df, FIG_DIR)
            print("✅ Generated comparison figures successfully")
        except Exception as e:
            print(f"⚠️ Error generating comparison figures: {str(e)}")
            
    # 3. Generate metric summaries
        print("\n🔄 Generating metric summaries...")
        metrics = ['RMSE', 'MAE', 'R²', 'MAPE %', 'SMAPE %']
        for metric in metrics:
            key = f'Validation {metric}'
            try:
                plt.figure(figsize=(12, 6))
                values = all_df[key].values if key in all_df.columns else all_df[metric].values
                plot_metric_comparison(
                    models=all_df['Model'].values,
                    values=values,
                    metric_name=metric,
                    save_path=FIG_DIR / f"comparisons/metric_{metric.lower().replace(' %', '').replace('²', '2')}.png"
                )
                print(f"✅ Generated {metric} comparison")
            except Exception as e:
                print(f"⚠️ Error generating {metric} comparison: {str(e)}")

    # Add SMAPE figure
        print(" ✓ Generating 06_smape_comparison.png...")
        fig, ax = plt.subplots(figsize=(14, 7))

        models = all_df['Model'].values
        smape_vals = all_df['Validation SMAPE %'].values if 'Validation SMAPE %' in all_df.columns else all_df['SMAPE'].values

        colors = ['#17a2b8' if m != 'CustomLSTM' else '#dc3545' for m in models]
        bars = ax.bar(models, smape_vals, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)

        for bar, val in zip(bars, smape_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%', ha='center', va='bottom')

        best_idx = np.argmin(smape_vals)
        bars[best_idx].set_linewidth(3)
        bars[best_idx].set_edgecolor('gold')

        ax.set_title('Model Comparison: SMAPE % (Lower is Better) ↓', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('SMAPE %', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        plt.xticks(rotation=45, ha='right')

        comp_dir = FIG_DIR / "comparisons"
        ensure_dir(comp_dir)
        save_fig(comp_dir / "06_smape_comparison.png")

        comparison = all_df
        best_model = comparison.iloc[0]

        # Generate LSTM timeline (cached model loaded above) if we successfully loaded it
        if lstm_model is not None and lstm_data is not None and y_lstm_pred is not None:
            try:
                generate_lstm_timeline(work, cutoff, lstm_data, y_lstm_pred, lstm_model, tab_scaler, lstm_scaler, FIG_DIR)
            except Exception as e:
                print(f"⚠️ Could not generate cached LSTM timeline: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(" ⚠️ Skipping LSTM timeline generation because cached LSTM model or data could not be loaded.")

    else:
        print("\n 🚀 Training models (no cache found or forced retrain)...\n")

        # EDA
        if not SKIP_EDA:
            eda_dir = perform_eda(train, OUT_ROOT)

        # TRAIN ML MODELS
        print_section("PHASE 1: Training 6 Traditional ML Models")

        ml_models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=RS),
            "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=2, random_state=RS),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=300, max_depth=12, n_jobs=2, random_state=RS),
            "GradientBoosting": GradientBoostingRegressor(learning_rate=0.05, n_estimators=600, max_depth=3, subsample=0.9, random_state=RS),
            "HistGradientBoosting": HistGradientBoostingRegressor(max_depth=12, learning_rate=0.05, max_iter=600, l2_regularization=1.0, random_state=RS),
        }

        ml_results = []
        ml_predictions = {}

        for name, model in ml_models.items():
            model_path = OUT_ROOT / f"{name}.joblib"
            if model_path.exists() and not RETRAIN_ML:
                print(f" ✓ {name:25s} → loaded from cache")
                model = joblib.load(model_path)
            else:
                print(f" ▶ {name:25s} → training...", end='', flush=True)
                model.fit(X_train, y_train)
                joblib.dump(model, model_path)
                print(" ✅")

            y_pred = model.predict(X_val)
            ml_predictions[name] = y_pred

            ml_results.append({
                "Model": name,
                "Validation RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred))),
                "Validation MAE": float(mean_absolute_error(y_val, y_pred)),
                "Validation R²": float(r2_score(y_val, y_pred)),
                "Validation MAPE %": safe_mape(y_val, y_pred),
        "Validation SMAPE %": smape(y_val, y_pred)
            })

        ml_df = pd.DataFrame(ml_results).sort_values("Validation RMSE").reset_index(drop=True)

        print(f"\n 📊 ML Models Performance:")
        print(ml_df.to_string(index=False))

        ml_df.to_csv(OUT_ROOT / "01_ml_models_comparison.csv", index=False)
        print(f" ✅ Saved: 01_ml_models_comparison.csv")

        # TRAIN LSTM
        print_section("PHASE 2: Training Improved Hybrid LSTM with Attention")

        lstm_data = create_hybrid_lstm_data(work, cutoff, FEATURES, lookback=LOOKBACK)

        lstm_scaler_path = OUT_ROOT / "lstm_scaler.joblib"
        if lstm_scaler_path.exists():
            lstm_scaler = joblib.load(lstm_scaler_path)
        else:
            lstm_scaler = StandardScaler()
            lstm_scaler.fit(lstm_data['y_train'].reshape(-1,1))
            joblib.dump(lstm_scaler, lstm_scaler_path)

        y_train_z = lstm_scaler.transform(lstm_data['y_train'].reshape(-1,1)).ravel()
        y_val_z = lstm_scaler.transform(lstm_data['y_val'].reshape(-1,1)).ravel()

        tab_scaler_path = OUT_ROOT / "tab_scaler.joblib"
        if tab_scaler_path.exists():
            tab_scaler = joblib.load(tab_scaler_path)
        else:
            tab_scaler = StandardScaler()
            tab_scaler.fit(lstm_data['X_train_tab'])
            joblib.dump(tab_scaler, tab_scaler_path)

        X_train_tab_z = tab_scaler.transform(lstm_data['X_train_tab'])
        X_val_tab_z = tab_scaler.transform(lstm_data['X_val_tab'])

        lstm_path = OUT_ROOT / "ImprovedHybridLSTMv13.keras"
        lstm_csvlog = LOG_DIR / "lstm_history.csv"

        if lstm_path.exists() and not RETRAIN_LSTM:
            print(f" ✓ ImprovedHybridLSTM → loaded from cache")
            lstm_model = keras_models.load_model(
                lstm_path,
                safe_mode=False,
                custom_objects={
                    'AttentionSqueeze': AttentionSqueeze,
                    'AttentionSum': AttentionSum
                }
            )
        else:
            print(f" ▶ ImprovedHybridLSTM → training...")
            lstm_model = build_improved_hybrid_lstm_with_attention(lookback=LOOKBACK, n_tabular_features=len(FEATURES))

            es = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            csv_logger = callbacks.CSVLogger(lstm_csvlog.as_posix(), append=False)

            lstm_model.fit(
                [lstm_data['X_train_seq'], X_train_tab_z],
                y_train_z,
                epochs=LSTM_EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=([lstm_data['X_val_seq'], X_val_tab_z], y_val_z),
                callbacks=[es, rl, csv_logger],
                verbose=1
            )

            lstm_model.save(lstm_path)

        y_lstm_pred_z = lstm_model.predict([lstm_data['X_val_seq'], X_val_tab_z], verbose=0).ravel()
        y_lstm_pred = lstm_scaler.inverse_transform(y_lstm_pred_z.reshape(-1,1)).ravel()

        lstm_results = {
            "Model": "ImprovedHybridLSTM",
            "Validation RMSE": float(np.sqrt(mean_squared_error(lstm_data['y_val'], y_lstm_pred))),
            "Validation MAE": float(mean_absolute_error(lstm_data['y_val'], y_lstm_pred)),
            "Validation R²": float(r2_score(lstm_data['y_val'], y_lstm_pred)),
            "Validation MAPE %": safe_mape(lstm_data['y_val'], y_lstm_pred),
            "Validation SMAPE %": smape(lstm_data['y_val'], y_lstm_pred)
        }

        lstm_df = pd.DataFrame([lstm_results])

        print(f"\n 📊 ImprovedHybridLSTM Performance:")
        print(lstm_df.to_string(index=False))

        lstm_df.to_csv(OUT_ROOT / "03_lstm_model_comparison.csv", index=False)
        print(f" ✅ Saved: 03_lstm_model_comparison.csv")

        # COMPARE & TIMELINES
        print_section("PHASE 3: Final Comparison & Timelines")

        comparison = pd.concat([ml_df, lstm_df], ignore_index=True).sort_values("Validation RMSE").reset_index(drop=True)

        print(f"\n 🏆 FINAL RANKING:\n")
        print(comparison.to_string(index=False))

        comparison.to_csv(OUT_ROOT / "02_all_models_comparison.csv", index=False)
        print(f" ✅ Saved: 02_all_models_comparison.csv")

        # ✅ GENERATE TIMELINES FOR BOTH ML & LSTM
        print_section("PHASE 4: Generating Timeline Visualizations")
        # ✅ GENERATE TIMELINES FOR BOTH ML & LSTM
        # Generate ML timelines (pass ml_models and X_val to the function)
        generate_ml_timelines(work, cutoff, y_val, ml_predictions, ml_models, X_val, FIG_DIR)

        # Generate LSTM timeline (correct arg order: lstm_model, tab_scaler, lstm_scaler, FIG_DIR)
        generate_lstm_timeline(work, cutoff, lstm_data, y_lstm_pred, lstm_model, tab_scaler, lstm_scaler, FIG_DIR)

        # ✅ GENERATE COMPARISON FIGURES
        generate_comparison_figures(ml_df, lstm_df, comparison, FIG_DIR)

        best_model = comparison.iloc[0]

    # SUMMARY
    print_section("SUMMARY")

    print(f"\n 🏆 BEST MODEL: {best_model['Model']}")
    print(f" Validation RMSE: {best_model.get('Validation RMSE', best_model.get('RMSE', 'N/A'))}")
    print(f" Validation MAE: {best_model.get('Validation MAE', best_model.get('MAE', 'N/A'))}")
    print(f" Validation R²: {best_model.get('Validation R²', best_model.get('R2', 'N/A'))}")
    print(f" Validation MAPE %: {best_model.get('Validation MAPE %', best_model.get('MAPE', 'N/A'))}%")
    print(f" Validation SMAPE %: {best_model.get('Validation SMAPE %', best_model.get('SMAPE', 'N/A'))}%")

    print(f"\n✅ Complete analysis saved to: {OUT_ROOT}")
    print(f"✅ CSV files location: {OUT_ROOT}/*.csv")
    print(f" • 01_ml_models_comparison.csv")
    print(f" • 02_all_models_comparison.csv")
    print(f" • 03_lstm_model_comparison.csv")
    print(f"✅ Comparison Figures ({FIG_DIR}/comparisons/):")
    print(f" • 01_rmse_comparison.png")
    print(f" • 02_mae_comparison.png")
    print(f" • 03_r2_comparison.png")
    print(f" • 04_mape_comparison.png")
    print(f" • 05_smape_comparison.png")
    print(f" • 06_multi_metric_comparison.png (6-in-1 dashboard)")
    print(f" • 04_mape_comparison.png")
    print(f" • 05_multi_metric_comparison.png (4-in-1 dashboard)")
    print(f" • 06_smape_comparison.png")

if __name__ == "__main__":
    main()

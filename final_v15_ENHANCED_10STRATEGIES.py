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
VAL_DAYS   = 60
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
    attention = layers.Lambda(lambda x: tf.keras.backend.squeeze(x, -1))(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(LSTM_UNITS * 2)(attention)
    attention = layers.Permute([2, 1])(attention)

    x_lstm_att = layers.Multiply()([x_lstm, attention])
    x_lstm_att = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(x_lstm_att)

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

def generate_ml_timelines(work, cutoff, y_val, ml_predictions, FIG_DIR):
    """Generate timelines for 6 ML models"""
    print("\n 📈 Generating timeline plots for 6 ML models...")

    val_indices = work["date"] > cutoff
    val_dates = work.loc[val_indices, "date"].values
    val_actual = work.loc[val_indices, "sales"].values
    val_store = work.loc[val_indices, "store"].values
    val_item = work.loc[val_indices, "item"].values

    print(f" ✓ Validation samples: {len(val_actual)}")

    # Find a common store-item pair
    store_id, item_id = 0, 0
    store_mask_ml = val_store == store_id
    item_mask_ml = val_item == store_id
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
            plt.figure(figsize=(16, 6))
            pred_values = ml_predictions[model_name][pair_mask_ml]
            actual_values = val_actual[pair_mask_ml]
            dates = val_dates[pair_mask_ml]

            if len(pred_values) != len(actual_values):
                min_len = min(len(pred_values), len(actual_values))
                pred_values = pred_values[:min_len]
                actual_values = actual_values[:min_len]
                dates = dates[:min_len]

            x_dates = np.arange(len(actual_values))
            x_labels = pd.to_datetime(dates).strftime("%Y-%m-%d").values

            plt.plot(x_dates, actual_values, label="Actual", linewidth=2.5,
                color='#2c3e50', marker='o', markersize=5, zorder=3)
            plt.plot(x_dates, pred_values, label=f"{model_name}", linewidth=2.5,
                color=colors[idx % len(colors)], alpha=0.85, marker='s',
                markersize=4, linestyle='--', zorder=2)

            plt.title(f"{model_name} — Store {store_id}, Item {item_id}",
                fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Date", fontsize=12, fontweight='bold')
            plt.ylabel("Units Sold", fontsize=12, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3, linestyle=':')

            ax = plt.gca()
            num_ticks = 12
            n = len(x_labels)
            tick_indices = np.linspace(0, n-1, min(num_ticks, n), dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([x_labels[i] for i in tick_indices], rotation=45, ha='right')

            mae = np.mean(np.abs(actual_values - pred_values))
            rmse = np.sqrt(np.mean((actual_values - pred_values)**2))
            stats_text = f"Actual μ={actual_values.mean():.1f}\nPred μ={pred_values.mean():.1f}\nMAE={mae:.2f}\nRMSE={rmse:.2f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontfamily='monospace')

            plt.tight_layout()
            timeline_dir = FIG_DIR / "timelines"
            ensure_dir(timeline_dir)
            filename = f"timeline_{model_name}_store_{store_id}__item_{item_id}.png"
            save_fig(timeline_dir / filename)
            print(f" ✓ {filename}")
            print(f" - Actual: mean={actual_values.mean():.2f}, std={actual_values.std():.2f}")
            print(f" - Pred: mean={pred_values.mean():.2f}, std={pred_values.std():.2f}")
            print(f" - MAE: {mae:.2f}, RMSE: {rmse:.2f}\n")

        except Exception as e:
            print(f" ❌ Error generating {model_name}: {str(e)}\n")
            plt.close()
            continue

# ============================================================================
# ✅ NEW: DEDICATED LSTM TIMELINE GENERATION
# ============================================================================

def generate_lstm_timeline(work, cutoff, lstm_data, y_lstm_pred, FIG_DIR):
    """Generate timeline ONLY for LSTM model"""
    print("\n 📈 Generating timeline plot for ImprovedHybridLSTM...")

    try:
        # Create validation mask for LSTM data
        val_indices = work["date"] > cutoff
        val_dates = work.loc[val_indices, "date"].values
        val_actual = work.loc[val_indices, "sales"].values
        val_store = work.loc[val_indices, "store"].values
        val_item = work.loc[val_indices, "item"].values

        # ✅ Use LSTM validation data directly (already has correct length)
        lstm_val_dates = val_dates[:len(lstm_data['y_val'])]
        lstm_val_store = val_store[:len(lstm_data['y_val'])]
        lstm_val_item = val_item[:len(lstm_data['y_val'])]

        # Find store-item pair
        store_id, item_id = 0, 0
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
        pred_values = y_lstm_pred[pair_mask]
        actual_values = lstm_data['y_val'][pair_mask]
        dates = lstm_val_dates[pair_mask]

        if len(pred_values) == 0 or len(actual_values) == 0:
            print(" ⚠️ No data found for this store-item pair!")
            return

        # Create plot
        plt.figure(figsize=(16, 6))
        x_dates = np.arange(len(actual_values))
        x_labels = pd.to_datetime(dates).strftime("%Y-%m-%d").values

        plt.plot(x_dates, actual_values, label="Actual", linewidth=2.5,
            color='#2c3e50', marker='o', markersize=5, zorder=3)
        plt.plot(x_dates, pred_values, label="ImprovedHybridLSTM", linewidth=2.5,
            color='#ee5a6f', alpha=0.85, marker='s', markersize=4, linestyle='--', zorder=2)

        plt.title(f"ImprovedHybridLSTM — Store {store_id}, Item {item_id}",
            fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Date", fontsize=12, fontweight='bold')
        plt.ylabel("Units Sold", fontsize=12, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, linestyle=':')

        ax = plt.gca()
        num_ticks = 12
        n = len(x_labels)
        tick_indices = np.linspace(0, n-1, min(num_ticks, n), dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([x_labels[i] for i in tick_indices], rotation=45, ha='right')

        mae = np.mean(np.abs(actual_values - pred_values))
        rmse = np.sqrt(np.mean((actual_values - pred_values)**2))
        stats_text = f"Actual μ={actual_values.mean():.1f}\nPred μ={pred_values.mean():.1f}\nMAE={mae:.2f}\nRMSE={rmse:.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontfamily='monospace')

        plt.tight_layout()
        timeline_dir = FIG_DIR / "timelines"
        ensure_dir(timeline_dir)
        filename = f"timeline_ImprovedHybridLSTM_store_{store_id}__item_{item_id}.png"
        save_fig(timeline_dir / filename)
        print(f" ✅ {filename} - SAVED SUCCESSFULLY")
        print(f" - Actual: mean={actual_values.mean():.2f}, std={actual_values.std():.2f}")
        print(f" - Pred: mean={pred_values.mean():.2f}, std={pred_values.std():.2f}")
        print(f" - MAE: {mae:.2f}, RMSE: {rmse:.2f}\n")

    except Exception as e:
        print(f" ❌ Error generating LSTM timeline: {str(e)}\n")
        import traceback
        traceback.print_exc()
        plt.close()

# ============================================================================
# ✅ NEW: COMPARISON FIGURES GENERATION
# ============================================================================

def generate_comparison_figures(ml_df, lstm_df, all_df, FIG_DIR):
    """Generate 5 comprehensive comparison figures"""

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
# MAIN EXECUTION
# ============================================================================

def main():
    print_section("COMPLETE DEMAND FORECASTING SYSTEM v13 WITH LSTM TIMELINE FIX + COMPARISON FIGURES")
    print(" EDA + CSV Caching + 6 ML Models + Improved Hybrid LSTM + Timelines + Comparisons")

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
                print(f" ✓ Loaded {name}")

        # Load LSTM model
        lstm_path = OUT_ROOT / "ImprovedHybridLSTMv13.keras"
        lstm_model = keras_models.load_model(lstm_path, safe_mode=False)

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

        # Generate timelines and comparisons
        generate_ml_timelines(work, cutoff, y_val, ml_predictions, FIG_DIR)
        generate_lstm_timeline(work, cutoff, lstm_data, y_lstm_pred, FIG_DIR)
        generate_comparison_figures(ml_df, lstm_df, all_df, FIG_DIR)

        comparison = all_df
        best_model = comparison.iloc[0]

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
            lstm_model = keras_models.load_model(lstm_path, safe_mode=False)
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

        generate_ml_timelines(work, cutoff, y_val, ml_predictions, FIG_DIR)
        generate_lstm_timeline(work, cutoff, lstm_data, y_lstm_pred, FIG_DIR)

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

    print(f"\n✅ Complete analysis saved to: {OUT_ROOT}")
    print(f"✅ CSV files location: {OUT_ROOT}/*.csv")
    print(f" • 01_ml_models_comparison.csv")
    print(f" • 02_all_models_comparison.csv")
    print(f" • 03_lstm_model_comparison.csv")
    print(f"✅ Visualizations: {FIG_DIR}/")
    print(f"✅ Timelines: {FIG_DIR}/timelines/")
    print(f" • ML Models: 6 timelines")
    print(f" • LSTM: 1 timeline")
    print(f"✅ Comparison Figures: {FIG_DIR}/comparisons/")
    print(f" • 01_rmse_comparison.png")
    print(f" • 02_mae_comparison.png")
    print(f" • 03_r2_comparison.png")
    print(f" • 04_mape_comparison.png")
    print(f" • 05_multi_metric_comparison.png (4-in-1 dashboard)")

if __name__ == "__main__":
    main()

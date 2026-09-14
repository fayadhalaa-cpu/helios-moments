import numpy as np
import pickle
from importlib.resources import files
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


def extract_features(v, fvals, f_b):
    """Create fixed-length features from v, f, and pointwise ratio f_b/f."""

    v = np.asarray(v, dtype=float)
    fvals = np.asarray(fvals, dtype=float)
    f_b = np.asarray(f_b, dtype=float)

    # remove noisy edges
    v = v[3:-1]
    fvals = fvals[3:-1]
    f_b = f_b[3:-1]

    # keep only finite values
    mask = np.isfinite(v) & np.isfinite(fvals) & np.isfinite(f_b)
    v = v[mask]
    fvals = fvals[mask]
    f_b = f_b[mask]

    if len(v) == 0:
        return np.full(14, np.nan)

    # avoid division by zero
    denom = np.where(np.abs(fvals) > 1e-12, fvals, np.nan)
    ratio = f_b / denom

    # keep only valid ratio points
    mask_ratio = np.isfinite(ratio)
    v_ratio = v[mask_ratio]
    ratio = ratio[mask_ratio]

    # if ratio becomes empty, handle safely
    if len(v_ratio) == 0:
        return np.full(14, np.nan)

    # weighted CDF
    f_sum = np.sum(fvals)
    if f_sum <= 0:
        f_sum = 1e-10

    cdf = np.cumsum(fvals) / f_sum

    # basic features
    v_mean = np.sum(v * fvals) / f_sum
    v_std = np.sqrt(np.sum(fvals * (v - v_mean) ** 2) / f_sum)

    # percentiles
    v_p10 = np.interp(0.10, cdf, v)
    v_p25 = np.interp(0.25, cdf, v)
    v_p50 = np.interp(0.50, cdf, v)
    v_p75 = np.interp(0.75, cdf, v)
    v_p80 = np.interp(0.80, cdf, v)
    v_p90 = np.interp(0.90, cdf, v)

    v_at_fmax = v[np.argmax(fvals)]

    ratio_p25 = np.interp(v_p25, v_ratio, ratio)
    ratio_p50 = np.interp(v_p50, v_ratio, ratio)
    ratio_p75 = np.interp(v_p75, v_ratio, ratio)
    ratio_p90 = np.interp(v_p90, v_ratio, ratio)
    ratio_fmax = np.interp(v_at_fmax, v_ratio, ratio)

    return np.array([
        v_mean,v_std,v_p50,v_p10,v_p25,v_p75,v_p80,v_p90,
        v_at_fmax,ratio_p25,ratio_p50,ratio_p75,ratio_p90,ratio_fmax], dtype=float)


def load_vcut_data(pkl_path=None):
    if pkl_path is None:
        pkl_path = files("helios_moments").joinpath(
            "training_data",
            "labeled_vdfs.pkl"
        )

    with pkl_path.open("rb") as handle:
        labeled_data = pickle.load(handle)

    X, y = [], []

    for item in labeled_data:
        v = item["v"]
        fvals = item["f"]
        f_b = item["f_b"]
        v_cut = item["v_cut"]

        X.append(extract_features(v, fvals, f_b))
        y.append(v_cut)

    return np.asarray(X), np.asarray(y)

def train_vcut_model(
    pkl_path=None,
    test_size=0.3,
    random_state=0,
    n_estimators=924,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=2,
    min_weight_fraction_leaf=0.0281464975975561,
    max_features="log2",
    bootstrap=True,
    max_samples=0.7244319636564938
):
    """
    Train a Random Forest model for v_cut prediction on one train/test split.

    Returns
    -------
    model : trained RandomForestRegressor
    X_train, X_test, y_train, y_test : arrays
        Train/test split used for fitting and evaluation.
    train_mse : float
        Mean squared error on the training set.
    test_mse : float
        Mean squared error on the test set.
    mae : float
        Mean absolute error on the test set.
    r2 : float
        R^2 score on the test set.
    mape : float
        Mean absolute percentage error on the test set.
    """

    # Load features and labels
    X, y = load_vcut_data(pkl_path)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Build model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion="squared_error",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        max_samples=max_samples if bootstrap else None,
        n_jobs=-1,
        random_state=random_state
    )

    # Fit
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_test, y_test_pred)

    return (
        model,
        X_train, X_test, y_train, y_test,
        train_mse, test_mse, mae, r2, mape
    )

# === --- Prediction function --- ===
def predict_v_cut(I1a, I1b, model):
    """Predict v_cut and estimate uncertainty from tree prediction spread."""

    v = I1a["v"].values
    fvals = I1a["df"].values

    v_b = I1b["v"].values
    f_b = I1b["df"].values

    # normalize I1a
    f_max_a = np.max(fvals) if np.max(fvals) != 0 else 1.0
    fvals = fvals / f_max_a

    # normalize I1b
    f_max_b = np.max(f_b) if np.max(f_b) != 0 else 1.0
    f_b= f_b / f_max_b

    # keep finite I1b points
    mask_b = np.isfinite(v_b) & np.isfinite(f_b)
    v_b = v_b[mask_b]
    f_b = f_b[mask_b]

    if len(v_b) < 2:
        return np.nan, np.nan

    # sort before interpolation
    idx = np.argsort(v_b)
    v_b = v_b[idx]
    f_b = f_b[idx]

    # interpolate I1b onto I1a grid
    f_b = np.interp(v, v_b, f_b)

    # extract new features
    features = np.array(extract_features(v, fvals, f_b), dtype=float).reshape(1, -1)

    if not np.all(np.isfinite(features)):
        return np.nan, np.nan

    # Get prediction from each tree
    tree_preds = np.array([tree.predict(features)[0] for tree in model.estimators_])
    v_cut_pred = tree_preds.mean()
    v_cut_std = tree_preds.std()

    return v_cut_pred, v_cut_std
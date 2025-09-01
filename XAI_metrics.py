import numpy as np
import pandas as pd


def calc_valid(list_cfs, model, preprocessor, features_num):
    valid_count = 0
    total = 0
    for X_org, X_cfs in list_cfs:
        # Remove the last row
        X_org = X_org.iloc[:, :features_num]
        X_cfs = X_cfs.iloc[:, :features_num]

        # Predict original class (X_org must be 2D)
        y_true = model.predict(preprocessor.transform(X_org), verbose=0).argmax(axis=1)[0]
        # Predict counterfactual class (batch)
        y_cfs = model.predict(preprocessor.transform(X_cfs), verbose=0).argmax(axis=1)


                # Count samples where class has changed
        valid_count += np.sum(y_cfs != y_true)
        total += len(y_cfs)

    return valid_count / total if total > 0 else 0


def calc_sparsity(list_cfs, categorical_features):
    """
    Automatically infer feature names and continuous feature value ranges, calculate the sum of normalized L1 sparsity for continuous and categorical features.
    :param list_cfs: list, each element is (X_org, X_cfs), X_org is original single sample DataFrame, X_cfs is counterfactual sample DataFrame
    :param preprocessor: used to maintain format consistency (not used for transformation here, only for parameter passing to ensure interface compatibility)
    :param categorical_features: list, categorical feature names
    :return: average sparsity
    """
    # Get all column names that have appeared
    all_cols = set()
    for X_org, X_cfs in list_cfs:
        all_cols.update(X_org.columns)
    feature_names = list(all_cols)
    # Infer continuous features
    cont_features = [f for f in feature_names if f not in categorical_features]
    # Calculate continuous feature value ranges (max-min after merging all samples in the list)
    df_all = pd.concat([X_org for X_org, X_cfs in list_cfs] +
                       [X_cfs for X_org, X_cfs in list_cfs], axis=0)
    feature_ranges = {f: (df_all[f].max() - df_all[f].min() if f in cont_features else 1) for f in feature_names}
    sparsities = []
    for X_org, X_cfs in list_cfs:
        x_org = X_org.iloc[0] if isinstance(X_org, pd.DataFrame) else X_org[0]
        for idx in range(X_cfs.shape[0]):
            x_cf = X_cfs.iloc[idx] if isinstance(X_cfs, pd.DataFrame) else X_cfs[idx]
            # Normalized L1 for continuous features
            cont_l1 = 0
            if cont_features:
                cont_diff = [abs(x_cf[f] - x_org[f]) / feature_ranges[f] if feature_ranges[f] != 0 else 0
                             for f in cont_features]
                cont_l1 = np.sum(cont_diff) / len(cont_features)
            # Categorical features
            cat_l1 = 0
            if categorical_features:
                cat_diff = [int(x_cf[f] != x_org[f]) for f in categorical_features]
                cat_l1 = np.sum(cat_diff) / len(categorical_features)
            sparsities.append(cont_l1 + cat_l1)
    return np.mean(sparsities)

def calc_continuous_proximity(list_cfs, continuous_features):
    """
    Continuous feature proximity: MAD normalized L1 norm
    """
    # Merge all data for MAD calculation
    df_all = pd.concat(
        [X_org for X_org, X_cfs in list_cfs] +
        [X_cfs for X_org, X_cfs in list_cfs], axis=0
    )
    # Calculate MAD for each continuous feature
    mad_dict = {}
    for f in continuous_features:
        mad_dict[f] = np.median(np.abs(df_all[f] - np.median(df_all[f])))
        if mad_dict[f] == 0:  # Prevent division by zero
            mad_dict[f] = 1

    proximities = []
    for X_org, X_cfs in list_cfs:
        x_org = X_org.iloc[0] if isinstance(X_org, pd.DataFrame) else X_org[0]
        for idx in range(X_cfs.shape[0]):
            x_cf = X_cfs.iloc[idx] if isinstance(X_cfs, pd.DataFrame) else X_cfs[idx]
            # L1 distance for continuous features
            cont_diffs = [
                abs(x_cf[f] - x_org[f]) / mad_dict[f] if mad_dict[f] != 0 else 0
                for f in continuous_features
            ]
            cont_prox = np.sum(cont_diffs) / len(continuous_features) if continuous_features else 0
            proximities.append(cont_prox)
    return np.mean(proximities)

def calc_categorical_proximity(list_cfs, categorical_features):
    """
    Categorical feature proximity: normalized L0 norm (mismatch ratio)
    """
    proximities = []
    for X_org, X_cfs in list_cfs:
        x_org = X_org.iloc[0] if isinstance(X_org, pd.DataFrame) else X_org[0]
        for idx in range(X_cfs.shape[0]):
            x_cf = X_cfs.iloc[idx] if isinstance(X_cfs, pd.DataFrame) else X_cfs[idx]
            # Number of inconsistent categorical features
            cat_diffs = [
                int(x_cf[f] != x_org[f])
                for f in categorical_features
            ]
            cat_prox = np.sum(cat_diffs) / len(categorical_features) if categorical_features else 0
            proximities.append(cat_prox)
    return np.mean(proximities)

def calc_manifold_distance(list_cfs, df, categorical_features):
    """
    Calculate manifold distance between counterfactuals and original dataset (1-NN), use Euclidean distance for continuous features and Hamming distance for categorical features.
    :param list_cfs: [(X_org, X_cfs)], X_org is original single sample df, X_cfs is counterfactual df
    :param df: original dataset DataFrame
    :param categorical_features: list, categorical feature names
    :return: average manifold distance
    """
    # Automatically infer feature order
    feature_names = list(df.columns[:-1])
    cont_features = [f for f in feature_names if f not in categorical_features]
    # Organize original data as numeric (categorical unified as str) for easy calculation
    df_ = df.copy()
    for f in categorical_features:
        df_[f] = df_[f].astype(str)
    X_ref = df_[feature_names].to_numpy()

    def compute_distance(x, y):
        # Euclidean distance for continuous features, Hamming distance for categorical features (0/1 then average)
        cont_dist = 0
        cat_dist = 0
        if cont_features:
            cont_dist = np.linalg.norm(np.array([x[feature_names.index(f)] for f in cont_features], dtype=float) -
                                      np.array([y[feature_names.index(f)] for f in cont_features], dtype=float))
        if categorical_features:
            cat_dist = np.mean([x[feature_names.index(f)] != y[feature_names.index(f)] for f in categorical_features])
        return cont_dist + cat_dist

    # 1-NN model with custom distance
    class CustomNN:
        def __init__(self, X_ref):
            self.X_ref = X_ref

        def kneighbors(self, X):
            dists = []
            for x in X:
                all_dists = [compute_distance(x, y) for y in self.X_ref]
                dists.append((np.min(all_dists), np.argmin(all_dists)))
            return np.array([[d[0]] for d in dists]), np.array([[d[1]] for d in dists])

    # Extract all counterfactuals
    cfs_all = []
    for _, X_cfs in list_cfs:
        df_cfs = X_cfs.copy()
        for f in categorical_features:
            df_cfs[f] = df_cfs[f].astype(str)
        cfs_all.append(df_cfs[feature_names].to_numpy())
    cfs_all = np.vstack(cfs_all)  # Merge all counterfactuals

    # Calculate distance from each counterfactual to nearest neighbor in original dataset
    custom_nn = CustomNN(X_ref)
    dists, _ = custom_nn.kneighbors(cfs_all)
    return float(np.mean(dists))


def check_constraint(list_cfs, 
                    immutable_features, 
                    nondecreasing_features):
    """
    Calculate constraint satisfaction rate for counterfactual sample set.
    :param list_cfs: [(X_org, X_cfs)], each item is original sample and its counterfactual (both single-row DataFrames)
    :param immutable_features: immutable feature name list
    :param nondecreasing_features: non-decreasing feature name list
    :return: average constraint satisfaction rate
    """
    # Constraint detection for single counterfactual
    def check_one(x_org, x_cf):
        # 1. Immutable features cannot be changed
        for f in immutable_features:
            if x_org[f].iloc[0] != x_cf[f].iloc[0]:
                return 0
        # 2. Non-decreasing features cannot decrease
        for f in nondecreasing_features:
            try:
                # Support type-safe comparison (e.g., int/float/str)
                if float(x_cf[f].iloc[0]) < float(x_org[f].iloc[0]):
                    return 0
            except:
                # If type doesn't support comparison, directly judge as failure
                return 0
        return 1

    # Iterate through all counterfactual samples
    results = []
    for X_org, X_cfs in list_cfs:
        # X_cfs may be multiple counterfactuals (multi-row DataFrame), check all
        for idx in range(len(X_cfs)):
            x_cf = X_cfs.iloc[[idx]]
            results.append(check_one(X_org, x_cf))
    return float(np.mean(results)) if results else 0.0

def calc_cf_num(list_cfs):
    """
    Calculate average number of counterfactuals generated per original sample
    :param list_cfs: [(X_org, X_cfs)]
    :return: average number (float)
    """
    if not list_cfs:
        return 0.0
    total = sum(len(X_cfs) for _, X_cfs in list_cfs)
    avg = total / len(list_cfs)
    return round(avg, 2)

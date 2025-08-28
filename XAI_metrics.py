import numpy as np
import pandas as pd


def calc_valid(list_cfs, model, preprocessor, features_num):
    valid_count = 0
    total = 0
    for X_org, X_cfs in list_cfs:
        # 去掉最后一行
        X_org = X_org.iloc[:, :features_num]
        X_cfs = X_cfs.iloc[:, :features_num]

        # 预测原始类别（X_org必须二维）
        y_true = model.predict(preprocessor.transform(X_org), verbose=0).argmax(axis=1)[0]
        # 预测反事实类别（批量）
        y_cfs = model.predict(preprocessor.transform(X_cfs), verbose=0).argmax(axis=1)


                # 统计类别发生变化的样本数
        valid_count += np.sum(y_cfs != y_true)
        total += len(y_cfs)

    return valid_count / total if total > 0 else 0


def calc_sparsity(list_cfs, categorical_features):
    """
    自动推断特征名和连续特征取值范围, 计算连续与类别特征的归一化L1稀疏度之和。
    :param list_cfs: list, 每个元素是 (X_org, X_cfs), X_org为原始单样本DataFrame, X_cfs为反事实样本DataFrame
    :param preprocessor: 用于保持格式一致（这里不作变换, 只用来传参以保证接口兼容）
    :param categorical_features: list, 类别特征名
    :return: 平均稀疏度
    """
    # 取所有出现过的列名
    all_cols = set()
    for X_org, X_cfs in list_cfs:
        all_cols.update(X_org.columns)
    feature_names = list(all_cols)
    # 推断连续特征
    cont_features = [f for f in feature_names if f not in categorical_features]
    # 计算连续特征的取值范围（全list所有样本合并后最大-最小）
    df_all = pd.concat([X_org for X_org, X_cfs in list_cfs] +
                       [X_cfs for X_org, X_cfs in list_cfs], axis=0)
    feature_ranges = {f: (df_all[f].max() - df_all[f].min() if f in cont_features else 1) for f in feature_names}
    sparsities = []
    for X_org, X_cfs in list_cfs:
        x_org = X_org.iloc[0] if isinstance(X_org, pd.DataFrame) else X_org[0]
        for idx in range(X_cfs.shape[0]):
            x_cf = X_cfs.iloc[idx] if isinstance(X_cfs, pd.DataFrame) else X_cfs[idx]
            # 连续特征归一化L1
            cont_l1 = 0
            if cont_features:
                cont_diff = [abs(x_cf[f] - x_org[f]) / feature_ranges[f] if feature_ranges[f] != 0 else 0
                             for f in cont_features]
                cont_l1 = np.sum(cont_diff) / len(cont_features)
            # 类别特征
            cat_l1 = 0
            if categorical_features:
                cat_diff = [int(x_cf[f] != x_org[f]) for f in categorical_features]
                cat_l1 = np.sum(cat_diff) / len(categorical_features)
            sparsities.append(cont_l1 + cat_l1)
    return np.mean(sparsities)

def calc_continuous_proximity(list_cfs, continuous_features):
    """
    连续特征接近度: MAD归一化L1范数
    """
    # 合并所有数据用于计算MAD
    df_all = pd.concat(
        [X_org for X_org, X_cfs in list_cfs] +
        [X_cfs for X_org, X_cfs in list_cfs], axis=0
    )
    # 计算每个连续特征的MAD
    mad_dict = {}
    for f in continuous_features:
        mad_dict[f] = np.median(np.abs(df_all[f] - np.median(df_all[f])))
        if mad_dict[f] == 0:  # 防止除0
            mad_dict[f] = 1

    proximities = []
    for X_org, X_cfs in list_cfs:
        x_org = X_org.iloc[0] if isinstance(X_org, pd.DataFrame) else X_org[0]
        for idx in range(X_cfs.shape[0]):
            x_cf = X_cfs.iloc[idx] if isinstance(X_cfs, pd.DataFrame) else X_cfs[idx]
            # 连续特征的L1距离
            cont_diffs = [
                abs(x_cf[f] - x_org[f]) / mad_dict[f] if mad_dict[f] != 0 else 0
                for f in continuous_features
            ]
            cont_prox = np.sum(cont_diffs) / len(continuous_features) if continuous_features else 0
            proximities.append(cont_prox)
    return np.mean(proximities)

def calc_categorical_proximity(list_cfs, categorical_features):
    """
    类别特征接近度: 归一化L0范数(错配比例)
    """
    proximities = []
    for X_org, X_cfs in list_cfs:
        x_org = X_org.iloc[0] if isinstance(X_org, pd.DataFrame) else X_org[0]
        for idx in range(X_cfs.shape[0]):
            x_cf = X_cfs.iloc[idx] if isinstance(X_cfs, pd.DataFrame) else X_cfs[idx]
            # 类别特征不一致数量
            cat_diffs = [
                int(x_cf[f] != x_org[f])
                for f in categorical_features
            ]
            cat_prox = np.sum(cat_diffs) / len(categorical_features) if categorical_features else 0
            proximities.append(cat_prox)
    return np.mean(proximities)

def calc_manifold_distance(list_cfs, df, categorical_features):
    """
    计算反事实与原始数据集的流形距离(1-NN), 连续特征用欧氏距离，类别特征用汉明距离。
    :param list_cfs: [(X_org, X_cfs)], X_org为原始单样本df, X_cfs为反事实df
    :param df: 原始数据集DataFrame
    :param categorical_features: list, 类别特征名
    :return: 平均流形距离
    """
    # 自动推断特征顺序
    feature_names = list(df.columns[:-1])
    cont_features = [f for f in feature_names if f not in categorical_features]
    # 整理原始数据为数值型（类别统一为str），方便计算
    df_ = df.copy()
    for f in categorical_features:
        df_[f] = df_[f].astype(str)
    X_ref = df_[feature_names].to_numpy()

    def compute_distance(x, y):
        # 连续特征欧氏距离，类别特征汉明距离（0/1再平均）
        cont_dist = 0
        cat_dist = 0
        if cont_features:
            cont_dist = np.linalg.norm(np.array([x[feature_names.index(f)] for f in cont_features], dtype=float) -
                                      np.array([y[feature_names.index(f)] for f in cont_features], dtype=float))
        if categorical_features:
            cat_dist = np.mean([x[feature_names.index(f)] != y[feature_names.index(f)] for f in categorical_features])
        return cont_dist + cat_dist

    # 1-NN模型自定义distance
    class CustomNN:
        def __init__(self, X_ref):
            self.X_ref = X_ref

        def kneighbors(self, X):
            dists = []
            for x in X:
                all_dists = [compute_distance(x, y) for y in self.X_ref]
                dists.append((np.min(all_dists), np.argmin(all_dists)))
            return np.array([[d[0]] for d in dists]), np.array([[d[1]] for d in dists])

    # 提取所有反事实
    cfs_all = []
    for _, X_cfs in list_cfs:
        df_cfs = X_cfs.copy()
        for f in categorical_features:
            df_cfs[f] = df_cfs[f].astype(str)
        cfs_all.append(df_cfs[feature_names].to_numpy())
    cfs_all = np.vstack(cfs_all)  # 所有反事实合并

    # 计算每个反事实与原始数据集中最近邻的距离
    custom_nn = CustomNN(X_ref)
    dists, _ = custom_nn.kneighbors(cfs_all)
    return float(np.mean(dists))


def check_constraint(list_cfs, 
                    immutable_features, 
                    nondecreasing_features):
    """
    计算反事实样本集合的约束满足率。
    :param list_cfs: [(X_org, X_cfs)], 每项为原始样本及其反事实(均为单行DataFrame)
    :param immutable_features: 不可变特征名list
    :param nondecreasing_features: 非递减特征名list
    :return: 平均约束满足率
    """
    # 用于单个反事实的约束检测
    def check_one(x_org, x_cf):
        # 1. 不可变特征不能改变
        for f in immutable_features:
            if x_org[f].iloc[0] != x_cf[f].iloc[0]:
                return 0
        # 2. 非递减特征不能减少
        for f in nondecreasing_features:
            try:
                # 支持类型安全地比较(比如int/float/str)
                if float(x_cf[f].iloc[0]) < float(x_org[f].iloc[0]):
                    return 0
            except:
                # 类型不支持比较就直接判失败
                return 0
        return 1

    # 遍历所有反事实样本
    results = []
    for X_org, X_cfs in list_cfs:
        # X_cfs 可能为多个反事实(多行DataFrame), 全部检测
        for idx in range(len(X_cfs)):
            x_cf = X_cfs.iloc[[idx]]
            results.append(check_one(X_org, x_cf))
    return float(np.mean(results)) if results else 0.0

def calc_cf_num(list_cfs):
    """
    计算平均每个原始样本生成的反事实个数
    :param list_cfs: [(X_org, X_cfs)]
    :return: 平均数量(float)
    """
    if not list_cfs:
        return 0.0
    total = sum(len(X_cfs) for _, X_cfs in list_cfs)
    avg = total / len(list_cfs)
    return round(avg, 2)

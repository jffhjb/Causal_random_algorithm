"""
测试脚本：验证因果反事实生成的可重现性
"""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from dowhy import gcm
from dowhy.gcm.auto import AssignmentQuality
import dice_ml

# 设置全局随机种子
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
try:
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
except Exception:
    pass

def set_seed(s: int = SEED):
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)

set_seed(SEED)

# 导入修改后的CausalRandom
import dice_causal_random
from dice_causal_random import CausalRandom

def make_random_dag_for_7(seed=42, p_x=0.35, p_to_y=0.5, min_parents_y=2):
    rng = np.random.default_rng(seed)
    Xs = [f"X{i}" for i in range(1, 7)]
    Y  = "Y"

    dag = nx.DiGraph()
    dag.add_nodes_from(Xs + [Y])

    # X 内部随机连边（只允许从索引小的指向大的）
    for i in range(len(Xs)):
        for j in range(i + 1, len(Xs)):
            if rng.random() < p_x:
                dag.add_edge(Xs[i], Xs[j])

    # Xi -> Y 的随机边
    parents_y = []
    for xi in Xs:
        if rng.random() < p_to_y:
            dag.add_edge(xi, Y)
            parents_y.append(xi)

    # 若父节点太少，强制补足
    if len(parents_y) < min_parents_y:
        need = min_parents_y - len(parents_y)
        candidates = [x for x in Xs if x not in parents_y]
        forced = rng.choice(candidates, size=need, replace=False).tolist()
        for xi in forced:
            dag.add_edge(xi, Y)

    # 确保是 DAG，且 Y 是汇点（没有子节点）
    assert nx.is_directed_acyclic_graph(dag)
    assert list(dag.successors(Y)) == []

    return dag

def simulate_sem(dag, n=1000, seed=42, nonlinear=False, noise_scale=1.0, binary_Y=True):
    rng = np.random.default_rng(seed)
    order = list(nx.topological_sort(dag))
    data = pd.DataFrame(index=range(n), columns=order, dtype=float)

    # 随机权重和偏置
    W, bias = {}, {}
    for u, v in dag.edges():
        W[(u, v)] = round(rng.uniform(-2.0, 2.0) * 0.8, 2)
    for v in dag.nodes():
        bias[v] = round(rng.uniform(-1.0, 1.0), 2)

    for v in order:
        parents = list(dag.predecessors(v))
        if not parents:  # root：标准正态
            data[v] = np.round(rng.normal(0, 1, size=n), 2)
        else:
            lin = bias[v]
            for p in parents:
                lin += W[(p, v)] * data[p].values

            if v == "Y" and binary_Y:
                # logistic 分类
                logits = lin + rng.normal(0, noise_scale, size=n)
                probs = 1 / (1 + np.exp(-logits))
                data[v] = (probs > 0.5).astype(int)
            else:
                if nonlinear:
                    nl = 0.6 * np.tanh(lin) + 0.3 * np.sin(lin)
                    val = np.round(lin + nl + rng.normal(0, noise_scale, size=n), 2)
                else:
                    val = np.round(lin + rng.normal(0, noise_scale, size=n), 2)
                data[v] = val

    return data

def make_7vars_dataset(n=1000, seed=42, p_x=0.35, p_to_y=0.5,
                       min_parents_y=2, nonlinear=False, noise_scale=1.0, binary_Y=True):
    dag = make_random_dag_for_7(seed=seed, p_x=p_x, p_to_y=p_to_y, min_parents_y=min_parents_y)
    df  = simulate_sem(dag, n=n, seed=seed, nonlinear=nonlinear,
                       noise_scale=noise_scale, binary_Y=binary_Y)
    cols = [f"X{i}" for i in range(1, 7)] + ["Y"]
    return dag, df[cols]

def build_simple_dnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(8, activation='relu', input_shape=(6,)))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def test_reproducibility():
    """测试可重现性"""
    print("开始测试可重现性...")
    
    # 生成数据集
    dag, df = make_7vars_dataset(n=500, seed=2025, nonlinear=True, binary_Y=True)
    
    # 准备数据
    target = df['Y']
    df_X = df.drop(columns=['Y'])
    dag.remove_node("Y")
    
    train_dataset, test_dataset, y_train, y_test = train_test_split(
        df, target, test_size=0.2, random_state=42, stratify=df['Y']
    )
    X_train_df = train_dataset.drop('Y', axis=1)
    X_test_df = test_dataset.drop('Y', axis=1)
    
    # 训练模型
    set_seed(1)
    model = build_simple_dnn()
    model.fit(X_train_df.values, to_categorical(y_train), epochs=10, batch_size=8, verbose=0)
    
    # 准备DiCE接口
    d = dice_ml.Data(
        dataframe=train_dataset,
        continuous_features=[c for c in train_dataset.columns if c != 'Y'],
        outcome_name='Y'
    )
    m = dice_ml.Model(model=model, backend="TF2")
    
    # 设置GCM模型
    np.random.seed(42)
    random.seed(42)
    scm = gcm.InvertibleStructuralCausalModel(dag)
    summary = gcm.auto.assign_causal_mechanisms(scm, df, quality=AssignmentQuality.GOOD)
    gcm.fit(scm, df)
    
    # 创建CausalRandom实例
    exp = CausalRandom(d, m, scm, random_seed=42)
    
    # 生成反事实
    e1 = exp.generate_counterfactuals(
        X_test_df[:5],  # 只测试5个样本以加快速度
        total_CFs=2,
        sample_size=100,
        random_seed=42
    )
    
    # 提取结果
    all_cfs = pd.concat([cf.final_cfs_df for cf in e1.cf_examples_list], ignore_index=True)
    
    # 计算summary_causal
    summary_evaluation = gcm.evaluate_causal_model(scm, all_cfs, compare_mechanism_baselines=True)
    
    return summary_evaluation

if __name__ == "__main__":
    # 运行两次测试
    print("第一次运行...")
    result1 = test_reproducibility()
    
    print("\n第二次运行...")
    result2 = test_reproducibility()
    
    # 比较结果
    print("\n=== 可重现性测试结果 ===")
    print(f"两次运行结果是否相同: {result1 == result2}")
    
    if result1 == result2:
        print("✅ 测试通过！结果完全可重现。")
    else:
        print("❌ 测试失败！结果不可重现。")
        print("\n第一次运行结果:")
        print(result1)
        print("\n第二次运行结果:")
        print(result2) 
"""
Test script: Verify the reproducibility of causal counterfactual generation
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

# Set global random seed
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

# Import modified CausalRandom
import dice_causal_random
from dice_causal_random import CausalRandom

def make_random_dag_for_7(seed=42, p_x=0.35, p_to_y=0.5, min_parents_y=2):
    rng = np.random.default_rng(seed)
    Xs = [f"X{i}" for i in range(1, 7)]
    Y  = "Y"

    dag = nx.DiGraph()
    dag.add_nodes_from(Xs + [Y])

    # Random edges within X (only allow edges from smaller to larger indices)
    for i in range(len(Xs)):
        for j in range(i + 1, len(Xs)):
            if rng.random() < p_x:
                dag.add_edge(Xs[i], Xs[j])

    # Random edges from Xi to Y
    parents_y = []
    for xi in Xs:
        if rng.random() < p_to_y:
            dag.add_edge(xi, Y)
            parents_y.append(xi)

    # If too few parent nodes, force additional ones
    if len(parents_y) < min_parents_y:
        need = min_parents_y - len(parents_y)
        candidates = [x for x in Xs if x not in parents_y]
        forced = rng.choice(candidates, size=need, replace=False).tolist()
        for xi in forced:
            dag.add_edge(xi, Y)

    # Ensure it's a DAG and Y is a sink (no child nodes)
    assert nx.is_directed_acyclic_graph(dag)
    assert list(dag.successors(Y)) == []

    return dag

def simulate_sem(dag, n=1000, seed=42, nonlinear=False, noise_scale=1.0, binary_Y=True):
    rng = np.random.default_rng(seed)
    order = list(nx.topological_sort(dag))
    data = pd.DataFrame(index=range(n), columns=order, dtype=float)

    # Random weights and biases
    W, bias = {}, {}
    for u, v in dag.edges():
        W[(u, v)] = round(rng.uniform(-2.0, 2.0) * 0.8, 2)
    for v in dag.nodes():
        bias[v] = round(rng.uniform(-1.0, 1.0), 2)

    for v in order:
        parents = list(dag.predecessors(v))
        if not parents:  # root: standard normal
            data[v] = np.round(rng.normal(0, 1, size=n), 2)
        else:
            lin = bias[v]
            for p in parents:
                lin += W[(p, v)] * data[p].values

            if v == "Y" and binary_Y:
                # logistic classification
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
    """Test reproducibility"""
    print("Starting reproducibility test...")
    
    # Generate dataset
    dag, df = make_7vars_dataset(n=500, seed=2025, nonlinear=True, binary_Y=True)
    
    # Prepare data
    target = df['Y']
    df_X = df.drop(columns=['Y'])
    dag.remove_node("Y")
    
    train_dataset, test_dataset, y_train, y_test = train_test_split(
        df, target, test_size=0.2, random_state=42, stratify=df['Y']
    )
    X_train_df = train_dataset.drop('Y', axis=1)
    X_test_df = test_dataset.drop('Y', axis=1)
    
    # Train model
    set_seed(1)
    model = build_simple_dnn()
    model.fit(X_train_df.values, to_categorical(y_train), epochs=10, batch_size=8, verbose=0)
    
    # Prepare DiCE interface
    d = dice_ml.Data(
        dataframe=train_dataset,
        continuous_features=[c for c in train_dataset.columns if c != 'Y'],
        outcome_name='Y'
    )
    m = dice_ml.Model(model=model, backend="TF2")
    
    # Set up GCM model
    np.random.seed(42)
    random.seed(42)
    scm = gcm.InvertibleStructuralCausalModel(dag)
    summary = gcm.auto.assign_causal_mechanisms(scm, df, quality=AssignmentQuality.GOOD)
    gcm.fit(scm, df)
    
    # Create CausalRandom instance
    exp = CausalRandom(d, m, scm, random_seed=42)
    
    # Generate counterfactuals
    e1 = exp.generate_counterfactuals(
        X_test_df[:5],  # Only test 5 samples to speed up
        total_CFs=2,
        sample_size=100,
        random_seed=42
    )
    
    # Extract results
    all_cfs = pd.concat([cf.final_cfs_df for cf in e1.cf_examples_list], ignore_index=True)
    
    # Calculate summary_causal
    summary_evaluation = gcm.evaluate_causal_model(scm, all_cfs, compare_mechanism_baselines=True)
    
    return summary_evaluation

if __name__ == "__main__":
    # Run test twice
    print("First run...")
    result1 = test_reproducibility()
    
    print("\nSecond run...")
    result2 = test_reproducibility()
    
    # Compare results
    print("\n=== Reproducibility Test Results ===")
    print(f"Are the results from two runs identical: {result1 == result2}")
    
    if result1 == result2:
        print("✅ Test passed! Results are completely reproducible.")
    else:
        print("❌ Test failed! Results are not reproducible.")
        print("\nFirst run results:")
        print(result1)
        print("\nSecond run results:")
        print(result2) 
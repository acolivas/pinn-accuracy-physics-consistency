# ==========================================================
# Representative 2D PINN Training Script
# ----------------------------------------------------------
# Purpose:
#   - Train a Physics-Informed Neural Network (PINN) for
#     2D steady flow reconstruction
#   - Predict: [magvel, xvel, yvel, press]
#     from inputs [x, y, invel]
#   - Total loss:
#       lambda_data * L_data
#     + lambda_ns   * L_ns
#     + lambda_bc   * L_bc
#
# Repository note:
#   - This is a simplified representative script provided
#     for repository demonstration.
#   - The configuration below uses a reduced sweep size and
#     short training duration for clarity and quick testing.
#   - Sweep ranges and epochs can be expanded for larger
#     experiment sets.
# ==========================================================

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


# ==========================================================
# 0) Reproducibility
# ==========================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================================
# 1) Data Loading
# ==========================================================
# Expected columns in the CSV:
#   Inputs:  x, y, invel
#   Outputs: magvel, xvel, yvel, press

DATA_PATH = os.path.join("data_sample", "sample_airflow_data.csv")
df = pd.read_csv(DATA_PATH)

input_vars = ["x", "y", "invel"]
output_vars = ["magvel", "xvel", "yvel", "press"]

required_cols = input_vars + output_vars
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")


# ==========================================================
# 2) Boundary Labeling (Inlet / Outlet / Walls / Interior)
# ==========================================================
# Geometry constants for the 2D baseline case
H = 3.0       # domain height
h = 0.168     # inlet height (top-left segment)
T = 0.48      # outlet height (bottom-right segment)
tol = 1e-5    # tolerance for boundary checks

xmin, xmax = df["x"].min(), df["x"].max()
ymin, ymax = df["y"].min(), df["y"].max()

df["boundary"] = "interior"
df["wall_type"] = None
df["wall_condition"] = None

# Inlet: top-left vertical edge segment
df.loc[
    (np.abs(df["x"] - xmin) < tol) &
    (df["y"] >= H - h) & (df["y"] <= H),
    "boundary"
] = "inlet"

# Outlet: bottom-right vertical edge segment
df.loc[
    (np.abs(df["x"] - xmax) < tol) &
    (df["y"] >= 0.0) & (df["y"] <= T),
    "boundary"
] = "outlet"

# Horizontal walls (top/bottom), excluding inlet/outlet
df.loc[
    ((np.abs(df["y"] - ymin) < tol) | (np.abs(df["y"] - ymax) < tol)) &
    (df["boundary"] == "interior"),
    ["boundary", "wall_type"]
] = ["wall", "horizontal"]

# Vertical walls (left/right), excluding inlet/outlet
df.loc[
    ((np.abs(df["x"] - xmin) < tol) | (np.abs(df["x"] - xmax) < tol)) &
    (df["boundary"] == "interior"),
    ["boundary", "wall_type"]
] = ["wall", "vertical"]

# Default wall condition
df.loc[df["boundary"] == "wall", "wall_condition"] = "no-slip"


# ==========================================================
# 3) Sampling and Train/Test Split Setup
# ==========================================================
DATA_FRACTION = 0.75
RANDOM_SEED = 42
TRAIN_RATIO = 0.80

df_sampled = df.sample(frac=DATA_FRACTION, random_state=RANDOM_SEED).reset_index(drop=True)

X = df_sampled[input_vars].values.astype(np.float32)
Y = df_sampled[output_vars].values.astype(np.float32)

# Boundary subsets for boundary condition loss
inlet_mask = df_sampled["boundary"] == "inlet"
outlet_mask = df_sampled["boundary"] == "outlet"
wall_mask = df_sampled["boundary"] == "wall"

X_inlet = df_sampled.loc[inlet_mask, input_vars].values.astype(np.float32)
Y_inlet = df_sampled.loc[inlet_mask, output_vars].values.astype(np.float32)

X_outlet = df_sampled.loc[outlet_mask, input_vars].values.astype(np.float32)
Y_outlet = df_sampled.loc[outlet_mask, output_vars].values.astype(np.float32)

X_wall = df_sampled.loc[wall_mask, input_vars].values.astype(np.float32)
wall_conditions = df_sampled.loc[wall_mask, "wall_condition"].values
wall_types = df_sampled.loc[wall_mask, "wall_type"].values


# ==========================================================
# 4) Model Definition
# ==========================================================
def build_pinn(neurons: int, num_layers: int, activation: str) -> tf.keras.Model:
    """
    Fully connected PINN.

    Inputs:
        [x, y, invel]

    Outputs:
        [magvel, xvel, yvel, press]
    """
    model = Sequential()
    model.add(Input(shape=(len(input_vars),)))

    for _ in range(num_layers):
        model.add(Dense(neurons, activation=activation))

    model.add(Dense(len(output_vars)))  # linear output layer
    return model


# ==========================================================
# 5) Loss Components
# ==========================================================
def wall_loss_by_condition(model, X_wall, wall_conditions, wall_types):
    """
    Wall boundary loss with optional condition handling.

    Conditions:
        - no-slip: enforce u=0 and v=0
        - free-slip (horizontal): enforce v=0
        - free-slip (vertical): enforce u=0
        - rough: scaled no-slip penalty
    """
    if len(X_wall) == 0:
        return tf.constant(0.0, dtype=tf.float32)

    pred_wall = model(tf.convert_to_tensor(X_wall))
    u = pred_wall[:, 1]  # xvel
    v = pred_wall[:, 2]  # yvel

    losses = []
    for i, condition in enumerate(wall_conditions):
        if condition == "no-slip":
            losses.append(tf.square(u[i]) + tf.square(v[i]))
        elif condition == "free-slip":
            if wall_types[i] == "horizontal":
                losses.append(tf.square(v[i]))
            elif wall_types[i] == "vertical":
                losses.append(tf.square(u[i]))
            else:
                losses.append(tf.square(u[i]) + tf.square(v[i]))
        elif condition == "rough":
            losses.append(0.1 * (tf.square(u[i]) + tf.square(v[i])))
        else:
            losses.append(tf.square(u[i]) + tf.square(v[i]))

    return tf.reduce_mean(losses) if losses else tf.constant(0.0, dtype=tf.float32)


def compute_loss_components(
    y_true,
    y_pred,
    model,
    X_f,
    lambda_data=1.0,
    lambda_ns=1.0,
    lambda_bc=1.0,
    nu=0.01,
    X_inlet=None,
    Y_inlet=None,
    X_outlet=None,
    Y_outlet=None,
    X_wall=None,
    wall_conditions=None,
    wall_types=None
):
    """
    Compute the following loss components:
        - Data loss: supervised MSE over available data
        - Physics loss: steady 2D incompressible Navier-Stokes residuals
        - BC loss: partial boundary supervision
    """
    X_f = tf.convert_to_tensor(X_f)
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    # Physics residuals via automatic differentiation
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(X_f)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(X_f)
            y_hat = model(X_f)
            _, u, v, p = tf.split(tf.cast(y_hat, tf.float32), 4, axis=1)

        du_dX = tape2.gradient(u, X_f)
        dv_dX = tape2.gradient(v, X_f)
        dp_dX = tape2.gradient(p, X_f)

    du_dx, du_dy = du_dX[:, 0:1], du_dX[:, 1:2]
    dv_dx, dv_dy = dv_dX[:, 0:1], dv_dX[:, 1:2]
    dp_dx, dp_dy = dp_dX[:, 0:1], dp_dX[:, 1:2]

    d2u = tape1.gradient(du_dx, X_f)
    d2u_y = tape1.gradient(du_dy, X_f)
    d2v = tape1.gradient(dv_dx, X_f)
    d2v_y = tape1.gradient(dv_dy, X_f)

    del tape1, tape2

    d2u = tf.zeros_like(X_f) if d2u is None else d2u
    d2u_y = tf.zeros_like(X_f) if d2u_y is None else d2u_y
    d2v = tf.zeros_like(X_f) if d2v is None else d2v
    d2v_y = tf.zeros_like(X_f) if d2v_y is None else d2v_y

    d2u_dx2 = d2u[:, 0:1]
    d2u_dy2 = d2u_y[:, 1:2]
    d2v_dx2 = d2v[:, 0:1]
    d2v_dy2 = d2v_y[:, 1:2]

    # Steady incompressible Navier-Stokes residuals (2D)
    momentum_u = u * du_dx + v * du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2)
    momentum_v = u * dv_dx + v * dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2)
    continuity = du_dx + dv_dy

    loss_u = tf.reduce_mean(tf.square(momentum_u))
    loss_v = tf.reduce_mean(tf.square(momentum_v))
    loss_cont = tf.reduce_mean(tf.square(continuity))
    physics_loss = loss_u + loss_v + loss_cont

    # Supervised data loss
    data_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Partial boundary condition loss
    bc_loss = tf.constant(0.0, dtype=tf.float32)

    # Inlet: supervise xvel (index 1)
    if X_inlet is not None and Y_inlet is not None and len(X_inlet) > 0:
        pred_inlet = model(tf.convert_to_tensor(X_inlet))[:, 1:2]
        targ_inlet = tf.convert_to_tensor(Y_inlet)[:, 1:2]
        bc_loss += tf.reduce_mean(tf.square(pred_inlet - targ_inlet))

    # Outlet: supervise pressure (index 3)
    if X_outlet is not None and Y_outlet is not None and len(X_outlet) > 0:
        pred_outlet = model(tf.convert_to_tensor(X_outlet))[:, 3:4]
        targ_outlet = tf.convert_to_tensor(Y_outlet)[:, 3:4]
        bc_loss += tf.reduce_mean(tf.square(pred_outlet - targ_outlet))

    # Walls
    if X_wall is not None and wall_conditions is not None and wall_types is not None and len(X_wall) > 0:
        bc_loss += wall_loss_by_condition(model, X_wall, wall_conditions, wall_types)

    total_loss = (
        lambda_data * data_loss +
        lambda_ns * physics_loss +
        lambda_bc * bc_loss
    )

    return data_loss, physics_loss, loss_u, loss_v, loss_cont, bc_loss, total_loss


# ==========================================================
# 6) Representative Sweep Configuration
# ==========================================================
# Reduced configuration for repository demonstration.
# Expand these lists for larger experiment sets.

lambda_data_list = [1.0]
lambda_ns_list = [1.0]
lambda_bc_list = [1.0]

neurons_list = [64]
num_layers_list = [3]

epochs_list = [10]  # reduced for demonstration; increase for full training runs
batch_sizes = [64]
learning_rates = [1e-3]
activation_list = ["tanh"]

NU = 0.01

results_rows = []
predictions_rows = []
run_id = 0

K.clear_session()


# ==========================================================
# 7) Training Loop
# ==========================================================
for neurons in neurons_list:
    for num_layers in num_layers_list:
        for activation in activation_list:
            for epochs in epochs_list:
                for batch_size in batch_sizes:
                    for learning_rate in learning_rates:
                        for lambda_data in lambda_data_list:
                            for lambda_ns in lambda_ns_list:
                                for lambda_bc in lambda_bc_list:

                                    run_id += 1
                                    print(
                                        f"\n[Run {run_id}] "
                                        f"neurons={neurons}, layers={num_layers}, act={activation}, "
                                        f"epochs={epochs}, bs={batch_size}, lr={learning_rate}, "
                                        f"lambda_data={lambda_data}, lambda_ns={lambda_ns}, lambda_bc={lambda_bc}"
                                    )

                                    X_train, X_test, Y_train, Y_test = train_test_split(
                                        X,
                                        Y,
                                        train_size=TRAIN_RATIO,
                                        random_state=RANDOM_SEED
                                    )

                                    model = build_pinn(neurons, num_layers, activation)
                                    optimizer = Adam(learning_rate=learning_rate)

                                    loss_total_list = []
                                    loss_data_list = []
                                    loss_phys_list = []
                                    loss_u_list = []
                                    loss_v_list = []
                                    loss_cont_list = []
                                    loss_bc_list = []
                                    r2_list = []
                                    epoch_time_list = []

                                    start_time = time.time()

                                    for ep in range(epochs):
                                        ep_start = time.time()

                                        with tf.GradientTape() as tape:
                                            Y_pred_train = model(X_train, training=True)
                                            (
                                                data_loss,
                                                phys_loss,
                                                lu,
                                                lv,
                                                lcont,
                                                bc_loss,
                                                total_loss
                                            ) = compute_loss_components(
                                                Y_train,
                                                Y_pred_train,
                                                model,
                                                X_train,
                                                lambda_data=lambda_data,
                                                lambda_ns=lambda_ns,
                                                lambda_bc=lambda_bc,
                                                nu=NU,
                                                X_inlet=X_inlet,
                                                Y_inlet=Y_inlet,
                                                X_outlet=X_outlet,
                                                Y_outlet=Y_outlet,
                                                X_wall=X_wall,
                                                wall_conditions=wall_conditions,
                                                wall_types=wall_types
                                            )

                                        grads = tape.gradient(total_loss, model.trainable_variables)
                                        optimizer.apply_gradients(zip(grads, model.trainable_variables))

                                        Y_pred_test = model(X_test, training=False).numpy()
                                        r2_val = r2_score(Y_test, Y_pred_test)

                                        loss_total_list.append(float(total_loss.numpy()))
                                        loss_data_list.append(float(data_loss.numpy()))
                                        loss_phys_list.append(float(phys_loss.numpy()))
                                        loss_u_list.append(float(lu.numpy()))
                                        loss_v_list.append(float(lv.numpy()))
                                        loss_cont_list.append(float(lcont.numpy()))
                                        loss_bc_list.append(float(bc_loss.numpy()))
                                        r2_list.append(float(r2_val))
                                        epoch_time_list.append(float(time.time() - ep_start))

                                        print(
                                            f"  epoch={ep:3d} | "
                                            f"loss={loss_total_list[-1]:.4e} | "
                                            f"R2={r2_val:.4f} | "
                                            f"data={loss_data_list[-1]:.4e} | "
                                            f"phys={loss_phys_list[-1]:.4e} | "
                                            f"bc={loss_bc_list[-1]:.4e}"
                                        )

                                    total_time_sec = time.time() - start_time

                                    mse = mean_squared_error(Y_test, Y_pred_test)
                                    mae = mean_absolute_error(Y_test, Y_pred_test)
                                    rmse = float(np.sqrt(mse))
                                    r2_final = r2_score(Y_test, Y_pred_test)

                                    magvel_pred = Y_pred_test[:, 0]
                                    u_pred = Y_pred_test[:, 1]
                                    v_pred = Y_pred_test[:, 2]
                                    magvel_comp = np.sqrt(u_pred**2 + v_pred**2)
                                    magvel_internal_error = float(
                                        np.mean(np.abs(magvel_pred - magvel_comp))
                                    )

                                    results_rows.append({
                                        "run_id": run_id,
                                        "neurons": neurons,
                                        "num_layers": num_layers,
                                        "activation": activation,
                                        "epochs": epochs,
                                        "batch_size": batch_size,
                                        "learning_rate": learning_rate,
                                        "train_ratio": TRAIN_RATIO,
                                        "data_fraction": DATA_FRACTION,
                                        "lambda_data": lambda_data,
                                        "lambda_ns": lambda_ns,
                                        "lambda_bc": lambda_bc,
                                        "nu": NU,
                                        "final_total_loss": loss_total_list[-1],
                                        "final_data_loss": loss_data_list[-1],
                                        "final_physics_loss": loss_phys_list[-1],
                                        "final_bc_loss": loss_bc_list[-1],
                                        "MSE": float(mse),
                                        "MAE": float(mae),
                                        "RMSE": float(rmse),
                                        "final_R2": float(r2_final),
                                        "training_time_sec": float(total_time_sec),
                                        "magvel_internal_error": magvel_internal_error,
                                    })

                                    df_pred = pd.DataFrame({
                                        "run_id": run_id,
                                        "x": X_test[:, 0],
                                        "y": X_test[:, 1],
                                        "invel": X_test[:, 2],
                                        "magvel": Y_pred_test[:, 0],
                                        "xvel": Y_pred_test[:, 1],
                                        "yvel": Y_pred_test[:, 2],
                                        "press": Y_pred_test[:, 3],
                                    })
                                    predictions_rows.append(df_pred)


# ==========================================================
# 8) Save Outputs
# ==========================================================
df_results = pd.DataFrame(results_rows)
df_predictions = (
    pd.concat(predictions_rows, ignore_index=True)
    if len(predictions_rows) > 0
    else pd.DataFrame()
)

RESULTS_CSV = "pinn_results.csv"
PREDICTIONS_CSV = "pinn_predictions.csv"

df_results.to_csv(RESULTS_CSV, index=False)
df_predictions.to_csv(PREDICTIONS_CSV, index=False)

print("\nPINN sweep complete.")
print(f"Saved: {RESULTS_CSV} ({len(df_results)} runs)")
print(f"Saved: {PREDICTIONS_CSV} ({len(df_predictions)} rows)")
print(df_results.head())
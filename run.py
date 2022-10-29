import numpy as np
from implementations import *

LOAD_DATA_MODE = {
    0: "DER",
    1: "PRI",
    2: "ALL",
}  # Allows us to switch between DER/PLI values or all

print("Loading the data")
y, x = load_data(train=True, mode=LOAD_DATA_MODE[0])  # Load data (DER)
y_indexes, x_test = load_data(train=False, mode=LOAD_DATA_MODE[0])

print("Preprocessing the data")
x_tr, x_te, y_tr, y_te = split_data(x, y, 0.8, np.random.seed())

x_tr = replace_min_999_by_col_mean(x_tr)  # Handle invalid values
x_te = replace_min_999_by_col_mean(x_te)

x_tr = build_poly_2(x_tr)  # Poly exp deg=2
x_te = build_poly_2(x_te)

x_tr, mean_x_tr, std_x_tr = standardize(x_tr)  # Standardize x
x_te, mean_x_te, std_x_te = standardize(x_te)

tx_tr = add_x_bias(x_tr)  # Add bias after normalisation to avoid NaNs
tx_te = add_x_bias(x_te)

print("Training the model")
# We run GD step times per epoch, for epochs epochs (same as running GD for epochs*step just lets us print intermediate results)
w, epochs, step, gamma = np.zeros(105), 100, 150, 1e-1

for i in range((int)(epochs)):
    w, loss_tr = logistic_regression(y_tr, tx_tr, w, step, gamma)
    loss_te = compute_log_loss(y_te, tx_te, w)
    print(f"Epoch {i} : Training loss: {loss_tr} Test loss: {loss_te}")

best_threshold, best_accuracy = threshold_selection_and_plot_log(
    tx_te, y_te, w
)  # Find the best threshold using grid search
print("best threshold=", best_threshold, "\nbest accuracy=", best_accuracy)

print("Preparing the output")
x_test = replace_min_999_by_col_mean(x_test)  # Handle invalid values

x_test = build_poly_2(x_test)  # Poly exp deg=2

x_test, mean_x_test, std_x_test = standardize(x_test)  # Standardize x

tx_test = add_x_bias(x_test)  # Add bias after normalisation to avoid NaNs

y_hat = build_prediction_log(
    tx_test, w, threshold=best_threshold, minus_one=True
)  # Compute the prediction and write it in a csv
write_to_csv(np.column_stack((y_indexes, y_hat)), "output.csv")
print("Done!")

from flask import Flask, render_template, Markup, request
#. venv/bin/activate

# Load data and remove outlier
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import csv
import tensorflow as tf
import matplotlib.pyplot as plt, mpld3


app = Flask(__name__)
@app.route('/')
def chart():
    #headers = {'Content-Type': 'text/html'}
    #return make_response(render_template("chart.html"),200,headers)
    return render_template('charts.htm')
    

@app.route('/drawChart', methods = ['GET', 'POST'])
def drawChart():
    if request.method == 'POST':
      df = pd.read_csv(request.files['file'])  # Change this to your directory
      
    # Shuffle to make sure the data is completely random and not provided in some order such as by geo location, etc.
    SEED = 456
    np.random.seed(SEED)
    df = df.iloc[np.random.permutation(len(df))]
    np.set_printoptions(suppress=True)
    NUM_VARS = 3
    N = len(df)
    print("Data size", N)

    # In[138]:
    X_params = ["HOST #", "CPU #", "DataBase SIZE TB"]
    X = df[X_params].astype(float)
    X = X.as_matrix()

    #  insert 1 to the matrix to calculate the +b value
    X = np.insert(X, NUM_VARS, 1, axis=1)

    # Expected output
    Y = df["Total Y Cost"].astype(float)
    # Y = np.array(Y).reshape(-1, 1)
    print(Y.shape)
    # In[141]:

    # ~2/3 for training
    n_trains = 130
    # ~1/3 for testing
    n_tests = len(df) - n_trains 

    #n_trains = 1
    # ~1/3 for testing
    #n_tests = 1
    X_train = X[:n_trains, :]
    Y_train = Y[:n_trains]
    print("X_train", len(X_train))

    X_test = X[-n_tests:, :]
    Y_test = Y[-n_tests:]
    print("X_test", len(X_test))

    # Shuffle with SEED = 456
    closed_form_rmse(X_train, Y_train, X_test, Y_test)
    fig = tf_multi_layers(X_train, Y_train, X_test, Y_test)
    plt.close()
    return mpld3.fig_to_html(fig)


def eval_rmse(X, Y, params):
    pred = X.dot(params)
    #  print("pred", pred)
    # print(pred)
    diff = pred - Y
    avg_cost = np.sqrt(np.dot(diff, diff) / len(X))
    return avg_cost


def eval_mae(X, Y, params):
    pred = X.dot(params)
    #  print("pred", pred)
    # print(pred)
    diff = pred - Y
    return np.mean(np.absolute(diff))


#  return avg_cost
def closed_form_rmse(X_train, Y_train, X_test, Y_test):
    # This is the solution that optimizes RMSE.
    sol = np.linalg.solve(X_train.T.dot(X_train), X_train.T.dot(Y_train))
    print("Closed form sol: ", sol)
    print("Closed form RMSE cost ", eval_rmse(X_test, Y_test, sol))
    print("Closed form MAE cost ", eval_mae(X_test, Y_test, sol))


def tf_multi_layers(X_train, Y_train, X_test, Y_test):
    SEED = 2017
    NUM_TRAINING_EPOCHS = 5000
    #  NUM_TRAINING_EPOCHS = 5000
    USE_MAE = True

    #   HIDDEN_NEURONS = [50,10]
    #   LEARNING_RATE = 0.0002
    #   REG = 0.1

    #  HIDDEN_NEURONS = [30]
    #  LEARNING_RATE = 0.0002
    #  REG = 0.1

    # No hidden layer. Simple linear model.
    HIDDEN_NEURONS = []
    LEARNING_RATE = 0.2
    REG = 0.0

    NUM_INPUT_COLS = X_train.shape[1]
    x = tf.placeholder(tf.float32, [None, NUM_INPUT_COLS])

    Ws = []
    Os = []
    previous_output = x
    previous_size = NUM_INPUT_COLS
    for current_size in HIDDEN_NEURONS:
        W = tf.Variable(tf.random_normal([previous_size, current_size], seed=SEED))
        O = tf.nn.relu(tf.matmul(previous_output, W))
        Os.append(O)
        Ws.append(W)
        previous_size = current_size
        previous_output = O

    Wf = tf.Variable(tf.random_normal([previous_size], seed=SEED))
    Ws.append(Wf)
    pred = tf.squeeze(tf.matmul(previous_output, tf.expand_dims(Wf, 1)))

    y = tf.placeholder(tf.float32, [None])
    diff = pred - y

    # RMSE cost function
    rmse_cost = tf.sqrt(tf.reduce_mean(tf.square(pred - y)))

    # MAE cost function.
    mae_cost = tf.reduce_mean(tf.abs(pred - y))

    # Change this parameter to define if optimization on RMSE or MAE cost
    if USE_MAE:
        total_cost = mae_cost
    else:
        total_cost = rmse_cost

    # Weight regularization to reduce overfitting.
    for W in Ws:
        total_cost += REG * tf.nn.l2_loss(W)

    training_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(total_cost)
    #  training_step = tf.train.AdamOptimizer(0.1).minimize(cost)

    # Training on first n_trains samples
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    plt.figure(1)

    train_costs = []
    test_costs = []
    steps = []
    for epoch in range(NUM_TRAINING_EPOCHS):
        sess.run(training_step, feed_dict={x: X_train, y: Y_train})
        # print(sess.run(tf.sqrt(cost), feed_dict={x:x_data, y:y_data}))

        if epoch % 100 == 0:
            output_train = sess.run(
                {"rmse_cost": rmse_cost,
                 "mae_cost": mae_cost,
                 "params": Wf,
                 "pred": pred,
                 "diff": diff,
                 },
                feed_dict={x: X_train, y: Y_train})
            output_test = sess.run(
                {"rmse_cost": rmse_cost,
                 "mae_cost": mae_cost,
                 "pred": pred,
                 "diff": diff,
                 },
                feed_dict={x: X_test, y: Y_test})
            # change here too for graph
            if USE_MAE:
                train_cost = output_train["mae_cost"]
                test_cost = output_test["mae_cost"]
            else:
                train_cost = output_train["rmse_cost"]
                test_cost = output_test["rmse_cost"]

            train_costs.append(train_cost)
            test_costs.append(test_cost)
            steps.append(epoch)

            print("#", epoch, "[RMSE] train cost:", output_train["rmse_cost"], "test cost:", output_test["rmse_cost"])
            print("#", epoch, "[MAE] train cost:", output_train["mae_cost"], "test cost:", output_test["mae_cost"])

    print("params", output_train["params"])

    #  plt.subplot(131)
    plot_costs(steps, train_costs, test_costs, USE_MAE)
    plt.autoscale(enable=False)
   
    #  plt.subplot(132)
    train_pred_plot, = plt.plot(Y_train, output_train["pred"], '+', color='b', label='Training set')

    #  plt.subplot(133)
    test_pred_plot, = plt.plot(Y_test, output_test["pred"], '+', color='r', label='Test set')
    max_cost = max(np.max(Y_train), np.max(Y_test))

    fig = plt.figure(1)
    plt.plot([1, max_cost], [1, max_cost], color='gray', linestyle='dotted')
    plt.xlabel('Real Total Cost')
    plt.ylabel('Predicted Total Cost')
    plt.legend(handles=[train_pred_plot, test_pred_plot])

    #  ylim = 30000
    #  plt.axis('equal')
    plt.ylim([0, max_cost])
    plt.xlim([0, max_cost])

    return fig
   


def plot_costs(steps, train_costs, test_costs, use_mae):
    fig = plt.figure(1)
    if use_mae:
        ylabel = 'Mean Absolute error'
    else:
        ylabel = 'Root Mean Square error'
    train_plot, = plt.plot(steps, train_costs, color='b', label='Training set')
    test_plot, = plt.plot(steps, test_costs, color='r', label='Test set')
    plt.xlabel('Training steps')
    plt.ylabel(ylabel)
    plt.legend(handles=[train_plot, test_plot])




if __name__ == "__main__":
    app.run(debug=True)




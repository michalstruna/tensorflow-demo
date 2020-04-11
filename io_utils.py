from argparse import ArgumentParser, ArgumentTypeError
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def value_float(x):
    try:
        x = float(x)
    except ValueError:
        raise ArgumentTypeError(f"Argument {x} is not a number.")

    if x < -1 or x > 1:
        raise ArgumentTypeError(f"Argument {x} is not from interval <-1, 1>.")

    return x


def read_args():
    parser = ArgumentParser()

    parser.add_argument('--run', '-r', type=value_float, nargs=3, metavar=('A', 'B', 'C'),
                        help='Calculate output for A, B and C input arguments.')

    parser.add_argument('--test', '-t', type=int, metavar='EPOCHS', help='Test neuron network.')

    parser.add_argument('--train', '-tr', type=float, nargs=3, metavar=('EPOCHS', 'BATCH_SIZE', 'VALIDATION_SPLIT'),
                        help='Train neuron network.')

    parser.add_argument('--loss', '-l', type=int, nargs=3, metavar=('START', 'END', 'STEP'),
                        help='Calculate loss error for hidden neurons countrs from START to END with STEP.')

    return parser.parse_args()


def print_result(inputs, result):
    print(f"a = {inputs[0]}\nb = {inputs[1]}\nc = {inputs[2]}")
    print(f"x = max(a, b, c) * b => {round(float(result[0][0]), 3)}")
    print(f"y = a^2 - b * c => {round(float(result[0][1]), 3)}")


def plot_test(inputs, outputs, goals):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    sc1 = ax.scatter3D(inputs[:, 0], inputs[:, 1], goals[:, 0], c='red')
    sc2 = ax.scatter3D(inputs[:, 0], inputs[:, 1], outputs[:, 0], c='blue')
    ax.scatter3D(inputs[:, 0], inputs[:, 1], goals[:, 1], c='red')
    ax.scatter3D(inputs[:, 0], inputs[:, 1], outputs[:, 1], c='blue')
    ax.legend([sc1, sc2], ['Očekávané', 'Skutečné'])

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('x, y')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter3D(inputs[:, 1], inputs[:, 2], goals[:, 0], c='red')
    ax.scatter3D(inputs[:, 1], inputs[:, 2], outputs[:, 0], c='blue')
    ax.scatter3D(inputs[:, 1], inputs[:, 2], goals[:, 1], c='red')
    ax.scatter3D(inputs[:, 1], inputs[:, 2], outputs[:, 1], c='blue')
    ax.set_xlabel('b')
    ax.set_ylabel('c')
    ax.set_zlabel('x, y')

    plt.legend()
    plt.show()


def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel('Epocha')
    plt.ylabel('Chyba')
    plt.grid()
    plt.show()


def plot_loss_range(loss, range):
    plt.plot(range, loss)
    plt.xlabel('Neuronů')
    plt.ylabel('Chyba')
    plt.grid()
    plt.show()

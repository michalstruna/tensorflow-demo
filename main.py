#!/usr/bin/python3

import numpy as np

from io_utils import read_args, print_result, plot_test, plot_loss, plot_loss_range
from model import Model

args = read_args()

if args.run:  # Calculate output for user input.
    model = Model(load=True)
    result = model.predict(np.array([args.run]))
    print_result(args.run, result)

elif args.test:  # Test neural network.
    model = Model(load=True)
    inputs, outputs, goals = model.test(args.test)
    plot_test(inputs, outputs, goals)

elif args.train:  # Train neural network.
    epochs, batch, validate = args.train
    model = Model()
    loss = model.train(int(round(epochs)), int(round(batch)), validate)
    model.save()
    plot_loss(loss)

elif args.loss:  # Calculate loss error for different numbers of neurons.
    start, end, step = args.loss
    epochs, batch, validate = 500, 32, 0.15
    errors = []

    for i in range(start, end, step):
        errors.append([])

        for j in range(10):
            model = Model(i)
            loss = model.train(int(round(epochs)), int(round(batch)), validate, 1)
            errors[-1].append(loss[-1])

        errors[-1] = sum(errors[-1]) / len(errors[-1])

    plot_loss_range(errors, range(start, end, step))

else:
    model = Model(load=True)
    model.summary()

import numpy as np
import re
import click
from matplotlib import pylab as plt


@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('pixel accuracy %')
    for i, log_file in enumerate(files):
        loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind = parse_log(log_file)
        disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, color_ind=i)
    plt.show()


def movingaverage(interval, window_size):
    if len(interval) == 0:
            return None

    #Padding the input array
    other = np.append(np.ones(int(window_size)-1, dtype=np.float32) * interval[1], interval);
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(other, window, 'valid')

def parse_log(log_file):
    with open(log_file, 'r') as log_file:
        log = log_file.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []

    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)
    print("Losses mean = " + str(np.mean(losses)))
    losses = movingaverage(losses, 100)
    if losses == None:
        print("Something has gone wrong for losses, moving averages returned None")
    else:
        print("Losses: " + str(losses))

    #accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* cerr = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)[\s\S]*?(cerr = )(?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"

    accuracies = []
    accuracy_iterations = []
    accuracies_iteration_checkpoints_ind = []

    for r in re.findall(accuracy_pattern, log):
        iteration = int(r[0])
        accuracy = float(100) - float(r[3]) * 100 / 94 / 94

        if iteration % 10000 == 0 and iteration > 0:
            accuracies_iteration_checkpoints_ind.append(len(accuracy_iterations))

        accuracy_iterations.append(iteration)
        accuracies.append(accuracy)

    accuracy_iterations = np.array(accuracy_iterations)
    print("Accuracies = " + str(accuracies))
    accuracies = np.array(accuracies)
    print("Accuracies mean = " + str(np.mean(accuracies)))
    accuracies = movingaverage(accuracies, 100)
    if accuracies == None:
        print("Something has gone wrong for accuracies, moving averages returned None")

    return loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind


def disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, color_ind=0):
    modula = len(plt.rcParams['axes.color_cycle'])
    ax1.plot(loss_iterations, losses, color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 0) % modula])
    ax2.plot(accuracy_iterations, accuracies, plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula])
    ax2.plot(accuracy_iterations[accuracies_iteration_checkpoints_ind], accuracies[accuracies_iteration_checkpoints_ind], 'o', color=plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula])


if __name__ == '__main__':
    main()

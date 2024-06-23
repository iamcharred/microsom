from numpy import (arange, tile)

def build_iteration_indexes(data_len, num_iterations,
                             random_generator, verbose=False):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training.

    If random_generator is not None, it must be an instance
    of numpy.random.RandomState and it will be used
    to randomize the order of the samples."""
    iterations_per_epoch = arange(data_len)
    if random_generator:
        random_generator.shuffle(iterations_per_epoch)
    iterations = tile(iterations_per_epoch, num_iterations)

    if verbose:
        print("total iterations ", iterations.shape[0])
        print("num epochs", num_iterations)

    return iterations
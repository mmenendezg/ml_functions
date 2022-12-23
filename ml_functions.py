import tensorflow as tf

_AUTOTUNE = tf.data.AUTOTUNE

def balanced_split(dataset, percentages=[0.80, 0.10, 0.10], verbose=False):
    """
    Split a given dataset into three datasets, defined by percentages, according to the classes in the dataset.

    Args:
        dataset (tf.data.Dataset): The dataset to be splitted.
        percentages (List[float], optional): A list with 3 elements that defines the percentage of the dataset
            that will be used for each of the sets. The elements must be floats in the range [0, 1] and
            must sum up to 1. Defaults to [0.80, 0.10, 0.10].
        verbose (bool, optional): A boolean that defines whether the function will print the split for each
            class and the final split. Defaults to False.

    Returns:
        tuple: A tuple with three datasets, corresponding to the training, validation and testing sets, respectively.
    """
    # Obtain the different classes in the datasets and sort the list
    list_classes = dataset.map(lambda x, y: y, num_parallel_calls=_AUTOTUNE).unique()
    list_classes = [class_.numpy() for class_ in list_classes]

    # Initialize the sets to False. This is just to avoid creating
    # a dataset without knowing the dimensions. This will not affect
    # the final dataset since the variable will be totally overwritten
    train_set = False
    valid_set = False
    test_set = False

    # Keep track of the total samples per set
    samples_train = 0
    samples_valid = 0
    samples_test = 0

    for class_ in list_classes:
        # Get the samples that match every class and samples per set
        tmp_dataset = dataset.filter(lambda x, y: y == class_)
        n_samples = len(list(tmp_dataset.as_numpy_iterator()))

        n_valid = int(percentages[1] * n_samples)
        n_test = int(percentages[1] * n_samples)
        n_train = n_samples - n_valid - n_test

        samples_train += n_train
        samples_valid += n_valid
        samples_test += n_test

        # Separate the sets and concatenate to the other classes sets
        tmp_train_set = tmp_dataset.take(n_train)
        tmp_valid_set = tmp_dataset.skip(n_train).take(n_valid)
        tmp_test_set = tmp_dataset.skip(n_train).skip(n_valid)

        train_set = (
            tmp_train_set
            if train_set == False
            else train_set.concatenate(tmp_train_set)
        )
        valid_set = (
            tmp_valid_set
            if valid_set == False
            else valid_set.concatenate(tmp_valid_set)
        )
        test_set = (
            tmp_test_set if test_set == False else test_set.concatenate(tmp_test_set)
        )

        if verbose == True:
            print(f"\tSplit for class {class_} is [{n_train}, {n_valid}, {n_test}]")

    if verbose == True:
        print("\nThe Split has been completed. The final split is the following: ")
        print(
            f"\tTraining Set: {samples_train}\n\tValidation Set:{samples_valid}\n\tTesting Set:{samples_valid}"
        )
    return train_set, valid_set, test_set
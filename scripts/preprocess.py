"""
Preprocessing script using the ExtendedDiscretizer (50 channels).
Used for initial data preparation before model training.
"""
from src.discretizer import ExtendedDiscretizer
from src.normalizer import Normalizer


def main():
    discretizer = ExtendedDiscretizer(
        timestep=0.8, store_masks=True,
        impute_strategy='previous', start_time='zero',
    )

    # Example usage:
    # data, header = discretizer.transform(X, header=header)
    # data, header = discretizer.transform_end_t_hours(X, los=los)
    # data, mask = discretizer.transform_reg(X, header=header)

    print("ExtendedDiscretizer initialized with {} channels".format(
        len(discretizer._id_to_channel)
    ))


if __name__ == '__main__':
    main()

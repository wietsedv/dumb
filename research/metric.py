import numpy as np


def scale(x: float):
    # return 1 - np.sqrt(1 - x)
    return 1 - (1 - x) ** (1/2)


if __name__ == "__main__":
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    target = [0.0, 0.10, 0.25, 0.50, 1.0]
    scaled = [scale(x) for x in scores]

    print(scores)
    print(target)
    print(scaled)

    print()
    print("1.00 - 0.99:", scale(1.0) - scale(0.99))
    print("0.60 - 0.50:", scale(0.6) - scale(0.5))
    print("0.10 - 0.00:", scale(0.1) - scale(0.0))

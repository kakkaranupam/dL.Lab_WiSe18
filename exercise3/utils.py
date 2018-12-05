import numpy as np

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


def one_hot_encoding(labels, num_classes):
    """Convert integer labels to one hot encoding.

    Example: y=[1, 2] --> [[0, 1, 0], [0, 0, 2]]
    """
    classes = [0, 1, 2, 3, 4]
    one_hot_labels = np.zeros(labels.shape + (num_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels


def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels


def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    gray = 2 * gray.astype('float32') - 1
    return gray


# def action_to_id_all(y):
#     l = r = s = a = b = 0
#     y = y.tolist()
#     y_new = [None] * len(y)
#     for i in range(len(y)):
#         print("y[i] : ", y[i], type(y[i]))
#         y_new[i] = action_to_id(y[i])
#         print("ORIG --- >>> ", y[i], " ||| A2I --- >>> ", y_new[i])
#         if (y_new[i] == LEFT):
#             l += 1
#         elif (y_new[i] == RIGHT):
#             r += 1
#         elif (y_new[i] == ACCELERATE):
#             a += 1
#         elif (y_new[i] == BRAKE):
#             print("BRAKE DETECTED!")
#             b += 1
#         else:
#             s += 1
#     print("RAW")
#     print("L --> ", l, " .. R --> ", r, " .. S --> ", s, " .. A --> ", a, \
#           " .. B --> ", b)
#     total = l + r + s + a + b
#     print("PERCENTAGE")
#     print("L --> ", l / total, " .. R --> ", r / total, " .. S --> ", s / total, \
#           " .. A --> ", a / total, " .. B --> ", b / total)
#     return np.array(y_new)


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    # if a[2]*10 == 2:
    # print("A22222222222222 : --- >>> ", type(a[2]))
    # print("BRAKEBRAKEBRAKE BC")
    if all(a == [-1.0, 0.0, 0.0]):
        return LEFT  # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]):
        return RIGHT  # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]):
        return ACCELERATE  # ACCELERATE: 3
    # elif all(a == [0.0, 0.0, 0.2]):
    elif a[2] > 0:
        # print("BC")
        return BRAKE  # BRAKE: 4
    else:
        return STRAIGHT  # STRAIGHT = 0

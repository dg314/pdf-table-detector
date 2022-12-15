import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_doc(image, bounding_box):
    fig, ax = plt.subplots()

    ax.imshow(image)

    # x, y = bounding_box[1], bounding_box[0]
    x, y = bounding_box[0], bounding_box[1]
    width, height = bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]

    rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    plt.show()
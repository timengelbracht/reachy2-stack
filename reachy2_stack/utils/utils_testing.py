import numpy as np
import matplotlib.pyplot as plt

def show_image(img: np.ndarray | None, title: str) -> None:
    if img is None:
        print(f"{title}: None (not displaying)")
        return

    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()
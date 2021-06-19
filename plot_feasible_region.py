import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

def plot_F(constraints, x_lim=-2, y_lim=6):

    d = np.linspace(x_lim, y_lim, 300)
    x,y = np.meshgrid(d,d)
    plt.imshow((
                ne.evaluate(constraints)
                ).astype(int), 
                extent=(x.min(), x.max(), y.min(), y.max()),
                origin="lower", cmap="Greys", alpha = 0.3)
    plt.show()

if __name__ == "__main__":
    constraints = """(
                        (0<=x)
                        & (x<=4)
                        & (0<=y)
                        & (y<=3)
                        & (-x+y<=2.5)
                        & (x+2*y<=9)

                    )"""

    plot_F(constraints)

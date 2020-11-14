import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")


def make_vtrain_video(cbfile, rpfile, thetafile, output, numx, numy):
    """Make video from the data in filename"""

    print('Loading data...')
    cbdata = pd.read_hdf(cbfile).values
    rpdata = pd.read_hdf(rpfile).values
    theta = pd.read_hdf(thetafile).values

    print('Data loaded. Making dictionary...')
    # Make dictionary
    cbvtrains = defaultdict(list)
    rpvtrains = defaultdict(list)
    for row in cbdata:
        cbvtrains[(row[0], row[1])] = row[2:]
    for row in rpdata:
        rpvtrains[(row[0], row[1])] = row[2:]

    print('Dictionary made. Writing video...')
    # Initiate writer
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Hydra Network', artist='Hengji Wang',
                    comment='Hydra neural network dynamics')
    writer = FFMpegWriter(fps=10, metadata=metadata)

    # Build canvas
    fig = plt.figure(figsize=(5, 10))

    # Traverse all time points and all neurons
    with writer.saving(fig, output, dpi=100):
        for t in tqdm(range(0, (len(row)-2), 1000)):
            plt.clf()
            # fig.patch.set_facecolor('b')
            # fig.patch.set_alpha(float((theta[t] - 3) / 2))
            ax = fig.add_subplot(111)
            ax.patch.set_facecolor('b')
            ax.patch.set_alpha(float((theta[t] - 3) / 4))
            ax.set_xlim(-0.5, numx - 0.5)
            ax.set_ylim(-0.5, numy - 0.5)
            ax.set_xticklabels(np.arange(numx), rotation=45, fontsize=8)
            ax.set_yticklabels(np.arange(numy), rotation=0, fontsize=8)
            plt.grid()
            for x, y in cbvtrains:
                ax.plot(x, y, color='green', marker='o', alpha=cbvtrains[(x, y)][t])
            for x, y in rpvtrains:
                ax.plot(x, y, color='red', marker='o', alpha=rpvtrains[(x, y)][t])
            writer.grab_frame()

    print('Video written.')





import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from torch import linalg as LA

def animate_sequence(sequence, nr_prediction_steps, sequence_gt=None, interval=500, figtitle='Animation', create_GIF=False):
    """animate_sequence

    Function that animates a given sequence of snapshots of a system.
    interval (ms) gives the time interval between to animation frames.
    """
    fig, ax = plt.subplots(nrows=sequence.shape[1], ncols=1 + (sequence_gt != None), squeeze=False, figsize=(13, 13))
    fig.suptitle(figtitle)
    artists = []
    artists_gt = []
    for i in range(sequence.shape[1]):
        if i == 2: 
            artists.append(ax[i, 0].imshow(sequence[0, i], cmap="coolwarm", vmin=1, vmax=2)) # plot the first of the sequence
            if sequence_gt != None:
                artists_gt.append(ax[i, 1].imshow(sequence_gt[0, i], cmap="coolwarm", vmin=1, vmax=2))    # plot in the right column the GT sequence, if given.  
        else:
            artists.append(ax[i, 0].imshow(sequence[0, i], cmap="coolwarm")) # plot the first of the sequence
            if sequence_gt != None:
                artists_gt.append(ax[i, 1].imshow(sequence_gt[0, i], cmap="coolwarm"))    # plot in the right column the GT sequence, if given.  


    ani = animation.FuncAnimation(fig=fig, func=partial(update, artists=artists, sequence=sequence, sequence_gt=sequence_gt, artists_gt=artists_gt, ax=ax), frames=nr_prediction_steps, interval=interval)
    return ani

# def func(frame, *fargs) -> iterable_of_artists
def update(frame, artists, sequence, ax, sequence_gt=None, artists_gt=None):
    # frame is just an index
    # update the AxesImage

    for i, artist in enumerate(artists):
        artist.set_data(sequence[frame, i])
        ax[i, 0].set_title(f'Pred Window {frame}, channel {i}, (norm: {LA.vector_norm(sequence[frame, i]):.2f})')
        if sequence_gt != None:
            artists_gt[i].set_data(sequence_gt[frame, i])
            ax[i, 1].set_title(f'GT Window {frame}, channel {i}, (norm: {LA.vector_norm(sequence_gt[frame, i]):.2f})')

    return artists
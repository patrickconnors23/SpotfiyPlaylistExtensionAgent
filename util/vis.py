import matplotlib
import matplotlib.pyplot as plt
import numpy as np


"""
Helper used to plot histogram
"""
def plotHist(ax, title, xlabel, ylabel, data):
    ax.hist(data, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

"""
Displays bar chart of most common keywords
used in playlist names
"""
def displayMostCommonKeyWord():
    count = 0
    pass
"""
do we need to know this?
"""
def displayPopularArtists():
    pass

"""
Displays bar chart of most common keywords
used in playlist names
"""
def displayMostCommonKeyWord():
    count = 0
    pass

"""
Displays histogram of playlist lengths,
allows us to get a sense of the size of
a typical playlist 
"""
def displayPlaylistLengthDistribution(df):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    numTracks = [len(x) for x in df.tracks]
    plotHist(
        ax=ax, 
        title= "Distribution of Number of Tracks per Playlist",
        xlabel="Number of Tracks",
        ylabel="Distribution",
        data=numTracks)
    plt.show()
    return

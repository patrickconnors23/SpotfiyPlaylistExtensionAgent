import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
Helper used to plot histogram
"""
def plotHist(ax, title, xlabel, ylabel, data):
    ax.hist(data, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

"""
Display the number of playlist inclusions for
X most popular artists
"""
def displayPopularArtists(df, lim=100):
    # Initialize dictionary of artist names
    artists = {}
    for playlist in df.tracks:
        for song in playlist:
            artsist = song["artist_name"]
            if artsist in artists:
                artists[artsist] += 1
            else: 
                artists[artsist] = 1
    
    # Sort artists by popularity
    sortedArtists = sorted(artists.items(), 
                           key=lambda x: x[1],
                           reverse=True)
    mostPopular = sortedArtists[:lim]
    artists, count = zip(*mostPopular)
    xvals = np.arange(len(mostPopular))
    # Plot data
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plt.bar(xvals, count)
    plt.xticks(
            ticks=xvals,
            labels=artists,
            rotation="vertical",
            fontsize=5
    )
    plt.xlabel("Artist")
    plt.ylabel("Number of Appearences")
    plt.title("Number of Playlist Appearences by Top 100 Artists")
    plt.savefig("figs/popularArtists.png")
    return

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
    plt.savefig("figs/playlistLengthDist.png")
    return

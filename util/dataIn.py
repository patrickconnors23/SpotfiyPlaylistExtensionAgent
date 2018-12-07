import json, display
import pandas as pd

def createCSVs(idx, numFiles, path, files):
    """
    Creates playlist and track csvs from
    json files
    """
    # Get correct number of files to work with
    files = files[idx:idx+numFiles]

    tracksSeen = set()
    playlistsLst = []
    trackLst = []

    for i, FILE in enumerate(files):
        if i % 10 == 0: print(i)
        # get full path to file
        name = path + FILE 
        with open(name) as f:
            data = json.load(f)
            playlists = data["playlists"]

            # for each playlist
            for playlist in playlists:
                for track in playlist["tracks"]:
                    if track["track_uri"] not in tracksSeen:
                        tracksSeen.add(track["track_uri"])
                        trackLst.append(track)
                playlist["tracks"] = [x["track_uri"] for x in playlist["tracks"]]
                playlistsLst.append(playlist)
    
    playlistDF = pd.DataFrame(playlistsLst)
    tracksDF = pd.DataFrame(trackLst)

    # Write DFs to CSVs
    print(f"Writing {len(playlistDF)} playlists to CSV")
    playlistDF.to_csv("data/CSV/playlists.csv")
    print(f"Writing {len(tracksDF)} tracks to CSV")
    tracksDF.to_csv("data/CSV/tracks.csv")
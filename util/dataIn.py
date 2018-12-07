import json, display
import pandas as pd

def createCSVs(idx, numFiles, path, files):
    """
    Creates playlist and track csvs from
    json files
    """
    def mapTracksToID(tracks):
        return [x["track_uri"] for x in tracks]
    
    def tracksToDFRow(track):
       pass 
    
    files = files[idx:idx+numFiles]

    tracksSeen = set()
    playlistsLst = []
    trackLst = []

    # for all files
    for i, FILE in enumerate(files):
        if i % 10 == 0: print(i)
        name = path + FILE 
        # Open file
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

    print(f"Writing {len(playlistDF)} playlists to CSV")
    playlistDF.to_csv("data/CSV/playlists.csv")
    print(f"Writing {len(tracksDF)} tracks to CSV")
    tracksDF.to_csv("data/CSV/tracks.csv")
# SpotfiyPlaylistExtensionAgent

Follow the below instructions to replicate our results.

## Setup & Installation

After cloning our repo, ensure that you have python=3.6.* installed.

Install the requirements with `pip3 install requirements.txt`

Next, [download](https://drive.google.com/file/d/1vvKVox1_MNezGJA7PCt_ZplQDqoVYA--/view) Spotify's Million Playlist Dataset (MPD). Unzip the file and place it in the project's root directory.

## Usage

To process the data from the MPD set, run `python3 main.py --parseData=x` where `x` is number of files to read (each file corresponds to 1000 playlists). We reccomend loading 100 files for local usage. This gives our classifier enough playlists to be meaningfully trained within a reasonable amount of time. 

After you have read the data from the MPD set you can simply run `python3 main.py` to run our classifier`

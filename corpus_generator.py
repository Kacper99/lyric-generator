import lyricsgenius as lg
import argparse
import json
import os

parser = argparse.ArgumentParser('Generate a corpus with all song lyrics from a specified artist')
parser.add_argument('-c', '--client', required=True)
parser.add_argument('-a', '--artist', required=False, metavar='Artist',
                    help='Name of the artist to generate corpus for', default=None)
parser.add_argument('-s', '--skip', required=False,
                    help='Skip downloading songs',
                    action='store_true', default=False)

args = parser.parse_args()

if args.artist is None:
    args.artist = str(input('Band name: '))

artist_file_name = args.artist.replace(' ', '_').lower()
lyrics_json_path = 'data/{}-lyrics.json'.format(artist_file_name)

if not args.skip:
    genius = lg.Genius(args.client)
    artist = genius.search_artist(args.artist)
    artist.save_lyrics('templyrics.json', sanitize=False)
    if os.path.exists(lyrics_json_path):
        os.remove(lyrics_json_path)
    os.rename('templyrics.json', lyrics_json_path)


all_song_lyrics = []
with open(lyrics_json_path) as f:
    lyrics = json.load(f)
    for s in lyrics['songs']:
        if s['lyrics'] is None:
            continue
        all_song_lyrics.append(s['lyrics'])

with open('data/' + artist_file_name + '-corpus.txt', 'w+', encoding='utf-8') as f:
    f.write('\n\n\n'.join(all_song_lyrics))


import lyricsgenius as lg
import argparse
import json

parser = argparse.ArgumentParser('Generate a corpus with all song lyrics from a specified artist')
parser.add_argument('-c', '--client', required=True)
parser.add_argument('-a', '--artist', required=True, metavar='Artist',
                    help='Name of the artist to generate corpus for')
parser.add_argument('-s', '--skip', required=False,
                    help='Skip downloading songs',
                    action='store_true', default=False)

args = parser.parse_args()

artist_file_name = args.artist.replace(' ', '_').lower()
lyrics_json_path = 'data/{}-lyrics.json'.format(artist_file_name)

if not args.skip:
    genius = lg.Genius(args.client)
    artist = genius.search_artist(args.artist, max_songs=3)
    artist.save_lyrics(lyrics_json_path, sanitize=False)

all_song_lyrics = []
with open(lyrics_json_path, encoding='utf-8') as f:
    lyrics = json.load(f)
    for s in lyrics['songs']:
        if s['lyrics'] is None:
            continue
        all_song_lyrics.append(s['lyrics'])

with open('data/' + artist_file_name + '-corpus.txt', 'w+') as f:
    f.write('\n\n\n'.join(all_song_lyrics))


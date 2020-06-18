import lyricsgenius as lg
import argparse

parser = argparse.ArgumentParser('Generate a corpus with all song lyrics from a specified artist')
parser.add_argument('-a', '--artist', required=True, metavar='Artist',
                    help='Name of the artist to generate corpus for')

args = parser.parse_args()

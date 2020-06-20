import statistics
import json

lengths = []
with open('data/lyrics.json', encoding='utf-8') as f:
    lyrics = json.load(f)
    for s in lyrics['songs']:
        if s['lyrics'] is None:
            continue
        lengths.append(len(s['lyrics']))

print(statistics.mean(lengths))
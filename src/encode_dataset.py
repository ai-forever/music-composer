import os
import torch
import joblib
import hashlib
import pretty_midi
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from lib import constants
from lib import midi_processing

DATA_DIR = 'test_dataset'
OUTPUT_DIR = 'encoded_dataset'
DS_FILE_PATH = './ds_files.pt' # path where ds_files.pt will be created

GENRES = ['classic', 'jazz', 'calm', 'pop']
MAX_LEN = 2048

print('creating dirs...')
[os.makedirs(OUTPUT_DIR+'/'+g, exist_ok=True) for g in GENRES]

print('collecting *.mid files...')
FILES = list(map(str, Path(DATA_DIR).rglob('*.mid')))

def encode_fn(i):
    """wrapper for loading i-th midi-file, encoding, padding and saving encoded tensor on disk"""
    file = FILES[i]
    max_len = MAX_LEN
    
    path, fname = os.path.split(file)
    try:
            midi = pretty_midi.PrettyMIDI(file)
            genre = path.split('/')[1]  # take GENRE from 'data/GENRE/xxx.mid'
    except:
        print(f'{i} not loaded')
        return -1
    
    assert genre in GENRES, f'{genre} is not in {GENRES}'
    
    fname, ext = os.path.splitext(fname)
    h = hashlib.md5(file.encode()).hexdigest()
    save_name = f'{OUTPUT_DIR}/{genre}/{fname}_{h}'
        
    events = midi_processing.encode(midi, use_piano_range=True)
    events = np.array(events)
    split_idxs = np.cumsum([max_len]*(events.shape[0]//max_len))
    splits = np.split(events, split_idxs, axis=0)
    n_last = splits[-1].shape[0]
    if n_last < 256:
        splits.pop(-1)
        drop_last = 1
    else:
        drop_last = 0
        
    for i, split in enumerate(splits):
        keep_idxs = midi_processing.filter_bad_note_offs(split)
        split = split[keep_idxs]
        eos_idx = min(max_len - 1, len(split))
        split = np.pad(split, [[0,max_len - len(split)]])
        split[eos_idx] = constants.TOKEN_END
        try:
            torch.save(split, f'{save_name}_{i}.pt')
        except OSError:  # if fname is too long
            save_name = f'{OUTPUT_DIR}/{genre}/{h}'
            torch.save(split, f'{save_name}_{i}.pt')
    return drop_last

cpu_count = joblib.cpu_count()
print(f'starting encoding in {cpu_count} processes...')
with ProcessPoolExecutor(cpu_count) as pool:
    x = list(tqdm(pool.map(encode_fn, range(len(FILES))), position=0, total=len(FILES)))

print('collecting encoded (*.pt) files...')
ds_files = list(map(str, Path(OUTPUT_DIR).rglob('*.pt')))
print('total encoded files:', len(ds_files))

torch.save(ds_files, DS_FILE_PATH)
print('ds_files.pt saved to', os.path.abspath(DS_FILE_PATH))
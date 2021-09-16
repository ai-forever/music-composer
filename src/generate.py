import os
import time
import torch
import argparse
import pretty_midi
import numpy as np
from tqdm import tqdm

from lib import constants
from lib import midi_processing
from lib import generation
from lib.midi_processing import PIANO_RANGE
from lib.model.transformer import MusicTransformer


def decode_and_write(generated, primer, genre, out_dir):
    '''Decodes event-based format to midi and writes resulting file to disk'''
    for i, (gen, g) in enumerate(zip(generated, genre)):
        midi = midi_processing.decode(gen)
        midi.write(f'{out_dir}/gen_{i:>02}_{id2genre[g]}.mid')

        
id2genre = {0:'classic',1:'jazz',2:'calm',3:'pop'}
genre2id = dict([[x[1],x[0]] for x in id2genre.items()])
tuned_params = {
    0: 1.1,
    1: 0.95,
    2: 0.9,
    3: 1.0
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--genre')
    parser.add_argument('--target_seq_length', default=512, type=int)
    parser.add_argument('--temperature', default=None, type=float)
    parser.add_argument('--topk', default=40, type=int)
    parser.add_argument('--topp', default=0.99, type=float)
    parser.add_argument('--topp_temperature', default=1.0, type=float)
    parser.add_argument('--at_least_k', default=1, type=int)
    parser.add_argument('--use_rp', action='store_true')
    parser.add_argument('--rp_penalty', default=0.05, type=int)
    parser.add_argument('--rp_restore_speed', default=0.7, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--keep_bad_generations', action='store_true')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--batch_size', default=8, type=int)
    args = parser.parse_args()


    try:
        genre_id = genre2id[args.genre]
    except KeyError:
        raise KeyError("Invalid genre name. Use one of ['classic', 'jazz', 'calm', 'pop']")

    load_path = args.load_path or '../checkpoints/model_big_v3_378k.pt'
    out_dir = args.out_dir or ('generated_' + time.strftime('%d-%m-%Y_%H-%M-%S'))
    batch_size = args.batch_size
    device = torch.device(args.device)
    remove_bad_generations = not args.keep_bad_generations

    default_params = dict(
        target_seq_length = 512,
        temperature = tuned_params[genre_id],
        topk = 40,
        topp = 0.99,
        topp_temperature = 1.0,
        at_least_k = 1,
        use_rp = False,
        rp_penalty = 0.05,
        rp_restore_speed = 0.7,
        seed = None,
    )

    params = {k:args.__dict__[k] if args.__dict__[k] else default_params[k] for k in default_params}

    os.makedirs(out_dir, exist_ok=True)

    # init model
    print('loading model...')
    model = MusicTransformer(device, n_layers=12, d_model=1024, dim_feedforward=2048, num_heads=16, vocab_size=constants.VOCAB_SIZE, rpr=True).to(device).eval()
    model.load_state_dict(torch.load(load_path, map_location=device))

    # add information about genre (first token)
    primer_genre = np.repeat([genre_id], batch_size)
    primer = torch.tensor(primer_genre)[:,None] + constants.VOCAB_SIZE - 4

    print('generating to:', os.path.abspath(out_dir))
    generated = generation.generate(model, primer, **params)
    generated = generation.post_process(generated, remove_bad_generations=remove_bad_generations)

    decode_and_write(generated, primer, primer_genre, out_dir)
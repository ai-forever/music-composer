import torch
import numpy as np
from tqdm import tqdm

from lib import midi_processing
from lib import constants
from lib.midi_processing import RANGES_SUM, get_type, NOTE_ON, NOTE_OFF
from lib.midi_processing import PIANO_RANGE


def generate(model, primer, target_seq_length=1024, temperature=1.0, topk=40, topp=0.99, topp_temperature=1.0, at_least_k=1, use_rp=False, rp_penalty=0.05, rp_restore_speed=0.7, seed=None, **forward_args):
    """
    Generate batch of samples, conditioned on `primer`. There are used several techniques for acquiring better generated samples such as:
    - temperature skewing for controlling entropy of distribuitions
    - top-k sampling
    - top-p (nucleus) sampling (https://arxiv.org/abs/1904.09751)
    - DynamicRepetitionPenaltyProcessor that prevents notes repeating
    values by default usualy are suitable for our models
        
    Parameters
    ----------
    model : MusicTransformer
        trained model.
    primer : torch.Tensor (B x N)
        primer for condition on.
        B = batch_size, N = seq_lenght.
        We are using the primer consisted of one token - genre. These tokens are {390:'classic', 391:'jazz', 392:'calm', 393:'pop'}.
    target_seq_length : int
        desired length  of generated sequences.
    temperature : float
        temperature alters the output distribuition of the model. Higher values ( > 1.0) lead to more stohastic sampling, lower values lead to more expected and predictable sequences (ending up with endlessly repeating musical patterns).
    topk : int
        restricts sampling from lower probabilities. It is the length of set of tokens from which sampling will be.
    topp : float
        restricts sampling from lower probabilities, but more adaptive then topk. see (https://arxiv.org/abs/1904.09751).
    topp_temperature : float
        temperature for counting cumulative sum doing topp sampling.
    at_least_k : int
        like topk, but force to sample from at least k tokens of higher probabilities.
    use_rp : bool
        use or not the DynamicRepetitionPenaltyProcessor (RP). Trying to prevent the generation of repeated notes.
    rp_penalty : float
        coef for RP. Higher values lead to more RP impact.
    rp_restore_speed : float
        how fast the penalty will be lifted. Lower values lead to more RP impact.
    seed : int
        fixes seed for deterministic generation.
    forward_args : dict
        args for model's forward.
        
    Returns
    -------
    generated : torch.Tensor (B x target_seq_length)
        generated batch of sequences.
    """
    device = model.device
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    if at_least_k < 1:
        at_least_k = 1
    B,N = primer.shape
    generated = torch.full((B,target_seq_length), constants.TOKEN_PAD, dtype=torch.int64, device=device)
    generated[..., :N] = primer.to(device)
    
    if use_rp:
        RP_processor = DynamicRepetitionPenaltyProcessor(B, penalty=rp_penalty, restore_speed=rp_restore_speed)
    whitelist_mask = make_whitelist_mask()
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(N, target_seq_length)):
            logits = model(generated[:, :i], **forward_args)[:, i-1, :]
            logits[:,~whitelist_mask] = float('-inf')
            p = torch.softmax(logits/topp_temperature, -1)
            
            # apply topk:
            if topk == 0:
                topk = p.shape[-1]
            p_topk, idxs = torch.topk(p, topk, -1, sorted=True)
            
            # apply topp:
            mask = p_topk.cumsum(-1) < topp
            mask[:,:at_least_k] = True
            logits_masked = logits.gather(-1, idxs)
            logits_masked[~mask] = float('-inf')
            p_topp = torch.softmax(logits_masked/temperature, -1)
            
            # apply penalty:
            if use_rp:
                p_penalized = RP_processor.apply_penalty(p_topp, idxs)
                ib = p_penalized.sum(-1) == 0
                if ib.sum() > 0:
                    # if all topp tokens get zeroes due RP_processor, then fallback to topk-sampling
                    p_fallback = p_topk[ib].clone()
                    p_fallback[mask[ib]] = 0.  # zeroing topp
                    p_penalized[ib] = p_fallback
                    
                ib = p_penalized.sum(-1) == 0
                if ib.sum() > 0:
                    # if topk tokens get zeroes, fallback to topp without RP
                    print('fallback-2')
                    p_penalized = p_topp
                p_topp = p_penalized
                    
            # sample:
            next_token = idxs.gather(-1, torch.multinomial(p_topp, 1))
            generated[:, i] = next_token.squeeze(-1)
            
            # update penalty:
            if use_rp:
                RP_processor.update(next_token)

    return generated[:, :i+1]


def post_process(generated, remove_bad_generations=True):
    """
    Post-process does 3 routines:
        1) removes long pauses (3+ seconds)
        2) clips velocities to range(30,100) to avoid dramaticly loud notes, which are not suitable for our case.
        3) removes bad generated samples. The model sometimes may generate music that consists only of many repeating notes. We try to detect them and remove from batch.
        
    Parameters
    ----------
    generated : torch.Tensor (B x N)
        batch of generated samples
        
    Returns
    -------
    filtered_generated : cleaner and slightly better sounding generated batch
    """
    generated = generated.cpu().numpy()
    remove_pauses(generated, 3)
    clip_velocity(generated)
    
    bad_filter = np.ones(len(generated), dtype=bool)
    
    if remove_bad_generations:
        for i, gen in enumerate(generated):
            midi = midi_processing.decode(gen)
            if detect_note_repetition(midi) > 0.9:
                bad_filter[i] = False

        if np.sum(bad_filter) != len(bad_filter):
            print(f'{np.sum(~bad_filter)} bad samples will be removed.')
        
    return generated[bad_filter]
    

def make_whitelist_mask():
    """Generate mask for PIANO_RANGE"""
    whitelist_mask = np.zeros(constants.VOCAB_SIZE, dtype=bool)
    whitelist_mask[PIANO_RANGE[0]:PIANO_RANGE[1]+1] = True
    whitelist_mask[128+PIANO_RANGE[0]:128+PIANO_RANGE[1]+1] = True
    whitelist_mask[128*2:] = True
    return whitelist_mask

    
class DynamicRepetitionPenaltyProcessor:
    """
    The class is trying to prevent cases where the model generates repetitive notes or musical patterns that degrade quality.
    It dynamically reduces and restores the probabilities of generatied notes.
    Each generated note will reduce its probability for the next step by `penalty` value (which is hyperparameter). If this note has been generated again, then we continue to reduce its probability, else we will gradually restore its probability (speed is controlled by restore_speed parameter).
    
    Parameters
    ----------
    bs : int
        batch_size. We need to know batch_size in advance to create the penalty_matrix.
    penalty : float
        value by which the probability will be reduced.
    restore_speed : float
        the number inversed to the number of seconds needs to fully restore probability from 0 to 1.
        for restore_speed equal to 1.0 we need 1.0 sec to restore, for 2.0 - 0.5 sec and so on.
    """
    def __init__(self, bs, penalty=0.3, restore_speed=1.0):
        self.bs = bs
        self.penalty = penalty
        self.restore_speed = restore_speed
        self.penalty_matrix = torch.ones(bs,128).to(device)
        
    def apply_penalty(self, p, idxs):
        p = p.clone()
        for b in range(len(p)):
            i = idxs[b]
            pi = p[b]
            mask = i < 128
            if len(i) > 0:
                pi[mask] = pi[mask]*self.penalty_matrix[b,i[mask]]
        return p
        
    def update(self, next_token):
        restoring = next_token - (128+128+32)  # only TS do restore
        restoring = torch.clamp(restoring.float(), 0, 100)/100*self.restore_speed
        self.penalty_matrix += restoring
        nt = next_token.squeeze(-1)
        nt = next_token[next_token < 128]
        self.penalty_matrix[:, nt] -= restoring + self.penalty
        torch.clamp(self.penalty_matrix, 0, 1.0, out=self.penalty_matrix)
        return restoring, nt
    

def detect_note_repetition(midi, threshold_sec=0.01):
    """
    Returns the fraction of note repetitions. Counts cases where prev_note_end == next_note_start at the same pitch ('glued' notes). Used in detection bad generated samples.
    
    Parameters
    ----------
    midi : prettyMIDI object
    threshold_sec : float
        intervals smaller then threshold_sec are treated as 'glued' notes.
    
    Returns
    -------
    fraction of notes repetitions relative to the number of all notes.
    """
    all_notes = [x for inst in midi.instruments for x in inst.notes if not inst.is_drum]
    if len(all_notes) == 0:
        return 0
    all_notes_np = np.array([[x.start,x.end,x.pitch,x.velocity] for x in all_notes])
    
    i_sort = np.lexsort([all_notes_np[:,0], all_notes_np[:,2]])

    s = []
    cur_p = -1
    cur_t = -1
    for t in all_notes_np[i_sort]:
        a,b,p,v = t
        if cur_p != p:
            cur_p = p
        else:
            s.append(a-cur_t)
        cur_t = b
    s = np.array(s)
    return (s < threshold_sec).sum()/len(s)


def remove_pauses(generated, threshold=3):
    """
    Fills  pauses by constants.TOKEN_PAD values. Only pauses that longer than `threshold` seconds are considered.
    Inplace operation. `generated` is a tensor (batch of sequences).
    
    Parameters
    ----------
    generated : torch.Tensor (B x N)
        generated batch of sequences.
    threshold : int/float
        the minimum seconds of silence to treat them as a pause.
    """
    mask = (generated>=RANGES_SUM[2]) & (generated<RANGES_SUM[3])
    seconds = ((generated-RANGES_SUM[2])+1)*0.01
    seconds[~mask] = 0

    res_ab = [[] for _ in range(seconds.shape[0])]

    for ib,i_seconds in enumerate(seconds):
        a,s = 0,0
        notes_down = np.zeros(128, dtype=bool)
        for i,(t,ev) in enumerate(zip(i_seconds,generated[ib])):
            typ = get_type(ev)
            if typ == NOTE_ON:
                pitch = ev
                notes_down[pitch] = True
            if typ == NOTE_OFF:
                pitch = ev-128
                notes_down[pitch] = False
                    
            if t == 0:
                if s >= threshold and notes_down.sum() == 0:
                    res_ab[ib].append([a,i,s])
                s = 0
                a = i+1
            s += t
        if s >= threshold and notes_down.sum() == 0:
            res_ab[ib].append([a,len(i_seconds),s])
    
    # remove inplace
    for ib,t in enumerate(res_ab):
        for a,b,s in t:
            generated[ib, a:b] = constants.TOKEN_PAD
            print(f'pause removed:',ib,f'n={b-a}',a,b,s)

        
def clip_velocity(generated, min_velocity=30, max_velocity=100):
    """
    Clip velocity to range(min_velocity, max_velocity). Since the model sometimes generate overloud sequences, we try to neutralize this effect.
    Inplace operation. `generated` is a tensor (batch of sequences).
    
    Parameters
    ----------
    generated : torch.Tensor (B x N)
        generated batch of sequences.
    min_velocity : int
    max_velocity : int
    """
    max_velocity_encoded = max_velocity*32//128 + RANGES_SUM[1]
    min_velocity_encoded = min_velocity*32//128 + RANGES_SUM[1]
    
    mask = (generated>=RANGES_SUM[1]) & (generated<RANGES_SUM[2])
    generated[mask] = np.clip(generated[mask], min_velocity_encoded, max_velocity_encoded)

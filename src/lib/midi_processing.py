import numpy as np
import pretty_midi


NOTE_ON = 0
NOTE_OFF = 1
SET_VELOCITY = 2
TIME_SHIFT = 3

MAX_TIME_SHIFT = 1.0
TIME_SHIFT_STEP = 0.01
RANGES = [128,128,32,100]

PIANO_RANGE = [21,96]  # 76 piano keys


def encode(midi, use_piano_range=True):
    """
    Encodes midi to event-based sequences for MusicTransformer.
    
    Parameters
    ----------
    midi : prettyMIDI object
        MIDI to encode.
    use_piano_range : bool
        if True, classical piano range will be used for skip pitches. Pitches which are not in range PIANO_RANGE will be skipped.
    
    Returns
    -------
    encoded_splits : list(list())
        splits of encoded sequences.
    """
    events = get_events(midi, use_piano_range=use_piano_range)
    if len(events) == 0:
        return []
    quantize_(events)
    add_time_shifts(events)
    encoded = encode_events(events)
    return encoded
    
    
def decode(encoded):
    """
    Decode event-based encoded sequence into MIDI object.
    
    Parameters
    ----------
    encoded : np.array or list
        encoded sequence to decode.
    
    Returns
    -------
    midi_out: PrettyMIDI object
        decoded MIDI.
    """
    midi_out = pretty_midi.PrettyMIDI()
    midi_out.instruments.append(pretty_midi.Instrument(0, name='piano'))
    notes = midi_out.instruments[0].notes
    
    notes_tmp = {}  # pitch: [vel, start, end]
    cur_time = 0
    cur_velocity = 100
    for ev in encoded:
        if ev < RANGES[0]:
            # NOTE_ON
            pitch = ev
            if notes_tmp.get(pitch) is None:
                notes_tmp[pitch] = [cur_velocity, cur_time]
        elif ev >= RANGES[0] and ev < sum(RANGES[:2]):
            # NOTE_OFF
            pitch = ev - RANGES[0]
            note = notes_tmp.get(pitch)
            if note is not None:  # check for overlaps (first-OFF mode)
                notes.append(pretty_midi.Note(note[0], pitch, note[1], cur_time))
                notes_tmp.pop(pitch)
        elif ev >= sum(RANGES[:2]) and ev < sum(RANGES[:3]):
            # SET_VELOCITY
            cur_velocity = max(1,(ev - sum(RANGES[:2]))*128//RANGES[2])
        elif ev >= sum(RANGES[:3]) and ev < sum(RANGES[:]):
            # TIME_SHIFT
            cur_time += (ev - sum(RANGES[:3]) + 1)*TIME_SHIFT_STEP
        else:
            continue

    for pitch, note in notes_tmp.items():
        if note[1] != cur_time:
            notes.append(pretty_midi.Note(note[0], pitch, note[1], cur_time))
        
    return midi_out


def round_step(x, step=0.01):
    return round(x/step)*step


def get_events(midi, use_piano_range=False):
    # helper function used in encode()
    # time, type, value
    events = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if use_piano_range and not (PIANO_RANGE[0] <= note.pitch <= PIANO_RANGE[1]):
                continue
            start = note.start
            end = note.end
            events.append([start, SET_VELOCITY, note.velocity])
            events.append([start, NOTE_ON, note.pitch])
            events.append([end, NOTE_OFF, note.pitch])
    events = sorted(events, key=lambda x: x[0])
    return events


def quantize_(events):
    for ev in events:
        ev[0] = round_step(ev[0])


def add_time_shifts(events):
    # populate time_shifts, helper function used in encode()
    times = np.array(list(zip(*events)))[0]
    diff = np.diff(times, prepend=0)
    idxs = diff.nonzero()[0]
    for i in reversed(idxs):
        if i == 0:
            continue
        t0 = events[i-1][0] # if i != 0 else 0
        t1 = events[i][0]
        dt = t1-t0
        events.insert(i, [t0, TIME_SHIFT, dt])


def encode_events(events):
    # helper function used in encode()
    out = []
    types = []
    for time, typ, value in events:
        offset = sum(RANGES[:typ])

        if typ == SET_VELOCITY:
            value = value*RANGES[SET_VELOCITY]//128
            out.append(offset+value)
            types.append(typ)

        elif typ == TIME_SHIFT:
            dt = value
            n = RANGES[TIME_SHIFT]
            enc = lambda x: int(x*n)-1
            for _ in range(int(dt//MAX_TIME_SHIFT)):
                out.append(offset+enc(MAX_TIME_SHIFT))
                types.append(typ)
            r = round_step(dt%MAX_TIME_SHIFT, TIME_SHIFT_STEP)
            if r > 0:
                out.append(offset+enc(r))
                types.append(typ)

        else:
            out.append(offset+value)
            types.append(typ)
            
    return out


RANGES_SUM = np.cumsum(RANGES)


def get_type(ev):
    if ev < RANGES_SUM[0]:
        # NOTE_ON
        return 0
    elif ev < RANGES_SUM[1]:
        # NOTE_OFF
        return 1
    elif ev < RANGES_SUM[2]:
        # VEL
        return 2
    elif ev < RANGES_SUM[3]:
        # TS
        return 3
    else:
        return -1

    
def filter_bad_note_offs(events):
    """Clear NOTE_OFF events for which the corresponding NOTE_ON event is missing."""
    notes_down = {}  # pitch: 1
    keep_idxs = set(range(len(events)))

    for i,ev in enumerate(events):
        typ = get_type(ev)

        if typ == NOTE_ON:
            pitch = ev
            notes_down[pitch] = 1
        if typ == NOTE_OFF:
            pitch = ev-128
            if notes_down.get(pitch) is None:
                # if NOTE_OFF without NOTE_ON, then remove the event
                keep_idxs.remove(i)
            else:
                notes_down.pop(pitch)
    
    return list(keep_idxs)
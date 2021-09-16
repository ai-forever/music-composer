import glob
import torch
import base64
import hashlib
import ipywidgets
import numpy as np
import lib.midi_processing
from midi2audio import FluidSynth
from torch.utils.data import Dataset
from IPython.display import Audio, display, FileLink, HTML


id2genre = {0:'classic', 1:'jazz', 2:'calm', 3:'pop'}
rugenre = {'classic': 'Классика', 'jazz': 'Джаз', 'calm': 'Эмбиент', 'pop': 'Поп'}
genre2id = dict([[x[1],x[0]] for x in id2genre.items()])
tuned_params = {
    0: 1.1,
    1: 0.95,
    2: 0.9,
    3: 1.0
}

def decode_and_write(generated, primer, genre, out_dir):
  '''Decodes midi files from event-based format and writes them to disk'''
  if len(glob.glob(out_dir + '/*.mid')) != 0:
    ids = [int(path.split('_')[-2]) for path in glob.glob(out_dir + '/*.mid')]
    start_from = max(ids)
  else:
    start_from = 0
    
  for i, (gen, g) in enumerate(zip(generated, genre)):
      midi = lib.midi_processing.decode(gen)
      midi.write(f'{out_dir}/gen_{i + start_from:>02}_{id2genre[g]}.mid')

def convert_midi_to_wav(midi_path):
    '''Converts MIDI to WAV format for listening in Colab'''
    FluidSynth("font.sf2").midi_to_audio(midi_path, midi_path.replace('.mid', '.wav'))

class DownloadButton(ipywidgets.Button):
    """Download button with dynamic content

    The content is generated using a callback when the button is clicked.
    """

    def __init__(self, filename: str, **kwargs):
        super(DownloadButton, self).__init__(**kwargs)
        self.filename = filename
        self.on_click(self.__on_click)

    def __on_click(self, b):
        with open(self.filename, 'rb') as f:
          b64 = base64.b64encode(f.read())
        payload = b64.decode()
        digest = hashlib.md5(self.filename.encode()).hexdigest()  # bypass browser cache
        id = f'dl_{digest}'

        display(HTML(f"""
<html>
<body>
<a id="{id}" download="{self.filename}" href="data:text/csv;base64,{payload}" download>
</a>

<script>
(function download() {{
document.getElementById('{id}').click();
}})()
</script>

</body>
</html>
"""))

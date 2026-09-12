"""Serial CPU inference timing; run after training has finished."""
from __future__ import annotations
import argparse
import platform
import statistics
import subprocess
import time
from pathlib import Path
import json
import torch
from corrected_experiments import ROOT, digest, generate, generate_stateful, load_corpus, save_json
from multi_dataset_fair_experiments import get_model_specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'artifacts/corrected_v1')
    args = parser.parse_args()
    torch.set_num_threads(1)
    factories = {name: factory for name, _, factory in get_model_specs()}
    (_, _, test), vocab, _ = load_corpus('tinyshakespeare')
    prompt = test[:64].unsqueeze(0)
    rows = []
    for path in sorted((args.results/'tinyshakespeare').glob('*.json')):
        result = json.loads(path.read_text())
        checkpoint = path.parent/result['checkpoint']
        if digest(checkpoint) != result['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        model = factories[result['model']](len(vocab))
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        modes = [('window64', generate)]
        if hasattr(model, 'backbone'):
            modes.append(('full_history_state', generate_stateful))
        for mode, fn in modes:
            fn(model, prompt, 128)
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                fn(model, prompt, 128)
                samples.append(time.perf_counter()-start)
            rows.append(dict(model=result['model'], seed=result['seed'], mode=mode,
                             seconds=samples, median_chars_per_second=128/statistics.median(samples),
                             checkpoint_sha256=result['checkpoint_sha256']))
    if len(rows) != 33:
        raise ValueError('Expected 18 fixed-window and 15 recurrent stateful measurements')
    try:
        hardware = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        hardware = platform.processor()
    save_json(args.results/'inference_timing.json', dict(rows=rows, hardware=hardware,
        platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
        threads=1, batch_size=1, prompt_length=64, continuation=128, warmups=1, repeats=3,
        scope='serial CPU wall time including prefill and token selection; no training speed claim'))
    print('Saved 33 serial inference measurements on', hardware)

if __name__ == '__main__':
    main()

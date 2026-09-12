"""Single-corpus entry point for the corrected protocol."""
import sys
from corrected_experiments import main
if __name__ == '__main__':
    if '--dataset' not in sys.argv:sys.argv.extend(['--dataset','tinyshakespeare'])
    main()

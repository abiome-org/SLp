"""Verify the bootstrap corpus and export the shared training partition contract."""
import argparse
from pathlib import Path
from data import Corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = Corpus(args.root)
    corpus.export_metadata(args.output)
    print({'genes': len(corpus.vocabulary) - 1, 'sources': [s.name for s in corpus.sources],
           'held_human': len(corpus.held_human), 'held_yeast': len(corpus.held_yeast)})

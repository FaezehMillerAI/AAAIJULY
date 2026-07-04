import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from nesy_gen.kg.primekg import build_radiology_cache_from_raw

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primekg-nodes", type=str, default="/kaggle/input/datasets/primekg/nodes.csv")
    parser.add_argument("--primekg-edges", type=str, default="/kaggle/input/datasets/primekg/kg.csv")
    parser.add_argument("--train-manifest", type=str, default="output/common_manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default="output/primekg_radiology_cache")
    parser.add_argument("--hops", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    
    nodes_path = Path(args.primekg_nodes)
    edges_path = Path(args.primekg_edges)
    train_manifest_path = Path(args.train_manifest)
    out_dir = Path(args.output_dir)
    
    print(f"Building radiology PrimeKG cache in {out_dir}...")
    build_radiology_cache_from_raw(
        primekg_nodes_path=nodes_path,
        primekg_edges_path=edges_path,
        train_manifest_path=train_manifest_path,
        output_dir=out_dir,
        hops=args.hops
    )
    print("PrimeKG cache build process completed successfully.")

if __name__ == "__main__":
    main()

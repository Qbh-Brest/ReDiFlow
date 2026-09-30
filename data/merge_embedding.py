import os
import torch
from argparse import ArgumentParser
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--esm_embeddings_path', type=str, default='D:\PythonProject medicine\project_mine\data/PDBBind_esm2_embeddings_raw',
                        help='Path to raw embeddings')
    parser.add_argument('--output_path', type=str, default='D:\PythonProject medicine\project_mine\data/PDBBind_esm2_embeddings.pt', help='Output file path')
    args = parser.parse_args()

    print(f"Start merging files from: {args.esm_embeddings_path}")

    files = [f for f in os.listdir(args.esm_embeddings_path) if f.endswith('.pt')]

    embedding_dict = {}
    for filename in tqdm(files):
        # Read single .pt file
        file_path = os.path.join(args.esm_embeddings_path, filename)
        data = torch.load(file_path)

        if 'representations' in data and 33 in data['representations']:
            rep = data['representations'][33]
        else:
            keys = list(data['representations'].keys())
            rep = data['representations'][keys[-1]]

        # Use filename (remove .pt suffix) as key
        key_name = filename.split('.')[0]
        embedding_dict[key_name] = rep

    print(f"Saving merged embeddings to {args.output_path} ...")
    torch.save(embedding_dict, args.output_path)
    print("Done! All embeddings merged successfully.")

import os
import torch
from argparse import ArgumentParser
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    # 注意：这里的 default 修改为你截图里的文件夹名字
    parser.add_argument('--esm_embeddings_path', type=str, default='D:\PythonProject medicine\project_mine\data/PDBBind_esm2_embeddings_raw',
                        help='Path to raw embeddings')
    parser.add_argument('--output_path', type=str, default='D:\PythonProject medicine\project_mine\data/PDBBind_esm2_embeddings.pt', help='Output file path')
    args = parser.parse_args()

    print(f"Start merging files from: {args.esm_embeddings_path}")

    # 获取文件列表
    files = [f for f in os.listdir(args.esm_embeddings_path) if f.endswith('.pt')]

    embedding_dict = {}
    for filename in tqdm(files):
        # 读取单个 .pt 文件
        file_path = os.path.join(args.esm_embeddings_path, filename)
        data = torch.load(file_path)

        # 提取第 33 层的表示（ESM-2 650M模型通常用第33层）
        # 注意：如果你用的不是 650M 模型，可能需要改这个数字，但在 DiffDock 等项目中通常是 33
        if 'representations' in data and 33 in data['representations']:
            rep = data['representations'][33]
        else:
            # 以此防备某些文件格式异常，如果找不到33层，尝试找最后一层或者直接报错
            keys = list(data['representations'].keys())
            rep = data['representations'][keys[-1]]

        # 以文件名（去掉.pt后缀）作为 key
        key_name = filename.split('.')[0]
        embedding_dict[key_name] = rep

    print(f"Saving merged embeddings to {args.output_path} ...")
    torch.save(embedding_dict, args.output_path)
    print("Done! All embeddings merged successfully.")

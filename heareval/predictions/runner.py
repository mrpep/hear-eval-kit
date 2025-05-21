#!/usr/bin/env python3
"""
Downstream training, using embeddings as input features and learning
predictions.
"""

import json
import random
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import click
import torch
from tqdm import tqdm

import heareval.gpu_max_mem as gpu_max_mem
from heareval.predictions.task_predictions import task_predictions

import shutil
import joblib

import pickle
import numpy as np

from sklearn.decomposition import PCA

# Cache this so the logger object isn't recreated,
# and we get accurate "relativeCreated" times.
_task_path_to_logger: Dict[Tuple[str, Path], logging.Logger] = {}


def get_logger(task_name: str, log_path: Path) -> logging.Logger:
    """Returns a task level logger"""
    global _task_path_to_logger
    if (task_name, log_path) not in _task_path_to_logger:
        logger = logging.getLogger(task_name)
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "predict - %(name)s - %(asctime)s - %(msecs)d - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)
        _task_path_to_logger[(task_name, log_path)] = logger
    return _task_path_to_logger[(task_name, log_path)]

def copy_stratified_file(src, dest, prop=0.5, lists_dir='/mnt/data/hear-learning-curve-lists'):
    src = Path(src)
    dest = Path(dest)
    if ('embeddings' in src.stem) and ('profile' not in src.stem) and ('test' not in src.stem) and ('valid' not in src.stem):
        with open(Path(lists_dir, src.parts[-2]+'-'+src.stem.split('.')[0]+'.list'), 'r') as f:
            lc_list = f.read().splitlines()
        ndim = np.memmap(src, dtype=np.float32, mode="r").shape[0]//len(lc_list)
        embedding_memmap = np.memmap(
            filename=dest,
            dtype=np.float32,
            mode="w+",
            shape=(len(lc_list), ndim),
        )
        try:
            for idx, xi in enumerate(lc_list):
                emb = np.load(Path(src.parent,src.stem.split('.')[0],f'{xi}.embedding.npy'))
                embedding_memmap[idx] = emb
        except:
            from IPython import embed; embed()
        embedding_memmap.flush()
    elif ('target-labels' in src.stem) and ('test' not in src.stem) and ('valid' not in src.stem):
        with open(src.with_name(src.stem.split('.')[0]+'.json'), 'r') as f:
            label_data = json.load(f)
        with open(Path(lists_dir, src.parts[-2]+'-'+src.stem.split('.')[0]+'.list'), 'r') as f:
            lc_list = f.read().splitlines()
        tl = [label_data[k] for k in lc_list]
        with open(dest, 'wb') as f:
            pickle.dump(tl, f)
    else:
        shutil.copy(src, dest)
        
class PCAModel:
    def __init__(self, num_components=None, layer_dims=None, variance_threshold=0.9):
        if num_components == 'dynamic':
            num_components = None
            self.dynamic = True
        elif num_components is None:
            num_components = None
        else:
            num_components = int(num_components)
        if layer_dims is not None:
            self.layer_dims = [int(x) for x in layer_dims.split(',')]
            self._models = [PCA(n_components=num_components) for i in range(len(self.layer_dims))]
        else:
            self._models = PCA(n_components=num_components)
        self.trained = False
        self.variance_threshold = variance_threshold
        
    def fit(self, x):
        idx_dim = 0
        for i,idx in enumerate(self.layer_dims):
            self._models[i].fit(x[:, idx_dim:idx_dim+idx])
            idx_dim += idx
        #explained_variances = [m.explained_variance_ratio_.sum() for m in self._models]
        if self.dynamic:
            num_components_dynamic = [np.argwhere(np.cumsum(m.explained_variance_ratio_)>self.variance_threshold)[0,0] for m in self._models]
            num_components = min(max(num_components_dynamic), min(self.layer_dims))
            self.components = num_components
        self.trained = True
        
    def transform(self, x):
        idx_dim = 0
        layer_components = []
        for i,idx in enumerate(self.layer_dims):
            layer_components.append(self._models[i].transform(x[:, idx_dim:idx_dim+idx]))
            idx_dim += idx
        if self.dynamic:
            layer_components = [c[:,:self.components] for c in layer_components]

        outs = np.transpose(np.array(layer_components),(1,0,2))

        return outs


@click.command()
@click.argument(
    "task_dirs",
    nargs=-1,
    required=True,
)
@click.option(
    "--grid-points",
    default=8,
    help="Number of grid points for randomized grid search "
    "model selection. (Default: 8)",
    type=click.INT,
)
@click.option(
    "--gpus",
    default=None if not torch.cuda.is_available() else "[0]",
    help='GPUs to use, as JSON string (default: "[0]" if any '
    "are available, none if not). "
    "See https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html#select-gpu-devices",  # noqa
    type=str,
)
@click.option(
    "--in-memory",
    default=True,
    help="Load embeddings in memory, or memmap them from disk. (Default: True)",
    type=click.BOOL,
)
@click.option(
    "--deterministic",
    default=True,
    help="Deterministic or non-deterministic. (Default: True)",
    type=click.BOOL,
)
@click.option(
    "--grid",
    default="default",
    help='Grid to use: ["default", "fast", "faster"]',
    type=str,
)
@click.option(
    "--shuffle",
    default=False,
    help="Shuffle tasks? (Default: False)",
    type=click.BOOL,
)
@click.option(
    "--seed",
    default=42,
    help="Seed for the experiment (Default: 42)",
    type=click.INT,
)
@click.option(
    "--dim_list",
    default=None,
    help="Choose specific dimensions of the features. (Default: None)",
    type=str
)
@click.option(
    "--apply_pca",
    default='false',
    help="Whether to apply PCA to the resulting embeddings. (Default: false)",
    type=str
)
@click.option(
    "--layer_dims",
    default='',
    help="When apply_pca=layerwise, dimensions of each layer. (Default:'')",
    type=str
)
@click.option(
    "--pca_dim",
    default=1024,
    help="PCA components to keep",
    type=str
)
@click.option(
    '--pca_file',
    default=None,
    help="Pretrained PCA model. (Default: None)",
    type=str
)
@click.option(
    '--train_prop',
    default=1.0,
    help="Downsample training sets. (Default: 1.0)",
    type=float
)
@click.option(
    '--subsample_type',
    default='balanced',
    help='Type of downsampling. (Default: balanced)',
    type=str
)

def runner(
    task_dirs: List[str],
    grid_points: int = 8,
    gpus: Any = None if not torch.cuda.is_available() else "[0]",
    in_memory: bool = True,
    deterministic: bool = True,
    grid: str = "default",
    shuffle: bool = False,
    seed: int = 42,
    apply_pca: str = 'false',
    layer_dims: str = '',
    pca_dim: Optional[int] = None,
    dim_list: Optional[str] = None,
    pca_file: Optional[str] = None,
    train_prop: float = 1.0,
    subsample_type: str = 'balanced'
    
) -> None:
    if gpus is not None:
        gpus = json.loads(gpus)

    if shuffle:
        random.shuffle(task_dirs)
    for task_dir in tqdm(task_dirs):
        task_path = Path(task_dir)
        task_path_ = task_path
        if dim_list is not None:
            dim_list_str=dim_list.replace('/','-').replace(':','-')
        else:
            dim_list_str=''
        if apply_pca != 'false':
            dim_list_str+='-pca'
            if pca_file:
                dim_list_str+=':{}'.format(pca_file.replace('/','-').replace('.','-'))
            if pca_dim:
                dim_list_str+=f'{pca_dim}'
        if train_prop < 1.0:
            train_prop_str = f'-prop-{train_prop}'
        elif train_prop == 1.0:
            train_prop_str = ''
        else:
            raise Exception('train_prop arg can not be larger than 1')
        task_path = task_path_.joinpath('seed{}{}{}'.format(seed,dim_list_str,train_prop_str))
        #Prepare seed path:
        task_path.mkdir(parents=True, exist_ok=True)
        files_to_cp = list(task_path_.glob('*.json')) + list(task_path_.glob('*.csv')) + list(task_path_.glob('*.pkl')) + list(task_path_.glob('*.npy'))
        for f in files_to_cp:
            if train_prop < 1.0:
                copy_stratified_file(f, Path(task_path, f.name), train_prop)
            else:
                shutil.copy(f, Path(task_path,f.name))

        if not task_path.is_dir():
            raise ValueError(f"{task_path} should be a directory")

        done_file = task_path.joinpath("prediction-done-seed{}.json".format(seed))
        if done_file.exists():
            # We already did this
            continue

        # Get embedding sizes for all splits/folds
        metadata = json.load(task_path.joinpath("task_metadata.json").open())

        log_path = task_path.joinpath("prediction.log")
        logger = get_logger(task_name=metadata["task_name"], log_path=log_path)

        logger.info(f"Computing predictions for {task_path.name}")
        embedding_sizes = []
        for split in metadata["splits"]:
            split_path = task_path.joinpath(f"{split}.embedding-dimensions.json")
            embedding_sizes.append(json.load(split_path.open())[-1])

        # Ensure all embedding sizes are the same across splits/folds
        embedding_size = embedding_sizes[0]
        if len(set(embedding_sizes)) != 1:
            raise ValueError("Embedding dimension mismatch among JSON files")

        if dim_list is not None:
            dim_file, dim_line = dim_list.split(':')
            with open(dim_file,'r') as f:
                dim_lines = f.read().splitlines()
                dim_map = {xi.split(':')[0]: [int(xii) for xii in xi.split(':')[1].split(',')] for xi in dim_lines}
                embedding_size = len(dim_map[dim_line])

        start = time.time()
        gpu_max_mem.reset()

        if pca_file is not None:
            pca_model = joblib.load(pca_file)
        else:
            pca_model = PCAModel(layer_dims=layer_dims, num_components=pca_dim)
        task_predictions(
            embedding_path=task_path,
            embedding_size=embedding_size,
            grid_points=grid_points,
            gpus=gpus,
            in_memory=in_memory,
            deterministic=deterministic,
            grid=grid,
            logger=logger,
            seed=seed,
            apply_pca=apply_pca,
            dim_list=dim_list,
            pca_model=pca_model,
            train_prop=train_prop,
            subsample_type=subsample_type
        )
        sys.stdout.flush()
        gpu_max_mem_used = gpu_max_mem.measure()
        logger.info(
            f"DONE took {time.time() - start} seconds to complete task_predictions"
            f"(embedding_path={task_path}, embedding_size={embedding_size}, "
            f"grid_points={grid_points}, gpus={gpus}, "
            f"gpu_max_mem_used={gpu_max_mem_used}, "
            f"gpu_device_name={gpu_max_mem.device_name()}, in_memory={in_memory}, "
            f"deterministic={deterministic}, grid={grid})"
        )
        sys.stdout.flush()
        open(done_file, "wt").write(
            json.dumps(
                {
                    "time": time.time() - start,
                    "embedding_path": str(task_path),
                    "embedding_size": embedding_size,
                    "grid_points": grid_points,
                    "gpus": gpus,
                    "gpu_max_mem": gpu_max_mem_used,
                    "gpu_device_name": gpu_max_mem.device_name(),
                    "in_memory": in_memory,
                    "deterministic": deterministic,
                    # "grid": grid
                },
                indent=4,
            )
        )


if __name__ == "__main__":
    runner()

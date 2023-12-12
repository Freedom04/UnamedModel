import sys
sys.path.append(".")
import torch
import torch.utils.data as data
import random
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix
import anndata

from config import Config


def read_mtx(mtx_file_path: str):
    """
    read sparse matrix file, such as .mtx file

    :param mtx_File_path:str, the path of the file need to be read

    return compressed sparse row matrix: scipy.sparse.csr_matrix
    """
    sparse_matrix = mmread(mtx_file_path)
    expression_matrix = sparse_matrix.A
    # (cells x genes) or (cells x peaks)
    # expression_matrix = expression_matrix.transpose((1, 0))
    # csr_expression_matrix = csr_matrix(expression_matrix, dtype=np.float64)

    # return a compressed sparse row matrix
    return expression_matrix

def initAnndata(mtx_path:str, features_path:str, barcodes_path:str):
    """
    Inititlize an anndata object from matrix.mtx, barcodes.tsv, and genes.tsv/peaks.tsv.

    return anndata obejct
    """
    matrix = read_mtx(mtx_path)
    features = pd.read_csv(features_path, sep='\t', header=None)
    barcodes = pd.read_csv(barcodes_path, sep='\t', header=None)

    adata = anndata.AnnData(X=matrix, 
                            obs=pd.DataFrame(index=barcodes[0]), 
                            var=pd.DataFrame(index=features[0])
                            )
    return adata

def initAndataWithtsv(counts_tsv_path:str):
    """
    Initilize an anndata object using counts.tsv file which has barcodes as column and genes as row
    
    return an anndata object
    """
    counts = pd.read_csv(counts_tsv_path, sep='\t', header=0)
    # csr_expression_matrix = csr_matrix(counts.values.transpose((1, 0)), dtype=np.float64)
    adata = anndata.AnnData(X=counts.values.transpose((1, 0)), 
                            var=pd.DataFrame(index=counts.index),
                            obs=pd.DataFrame(index=counts.columns)
                            )
    return adata



class Dataloader(data.Dataset):
    def __init__(self, train = True, rna_expression_matrix = None, atac_expression_matrix:np.ndarray = None):
        """
        :param: barcodes: np.ndarray, barcodes to label cells
        :param: expression_matrix: csr_matrix, rna expression matrix or atac expression matrix
        :param: feature: np.ndarray, rna genes or atac peaks
        """
        self.train = train
        self.rna_expression_matrix = rna_expression_matrix
        self.atac_expression_matrix = atac_expression_matrix    
        self.rna_input_size = self.rna_expression_matrix.shape[1]
        self.sample_num = self.rna_expression_matrix.shape[0]
        self.atac_input_size = self.atac_expression_matrix.shape[1]

    def __getitem__(self, index):
        if self.train:        
            rand_idx = random.randint(0, self.sample_num - 1)
            rna_sample = np.array(self.rna_expression_matrix[rand_idx].todense())
            atac_sample = np.array(self.atac_expression_matrix[rand_idx].todense())
            rna_sample = rna_sample.reshape((-1, self.rna_input_size))
            atac_sample = atac_sample.reshape((-1, self.atac_input_size))
 
            return rna_sample, atac_sample

        else:
            rna_sample = np.array(self.rna_expression_matrix[index].todense())
            atac_sample = np.array(self.atac_expression_matrix[index].todense())

            rna_sample = rna_sample.reshape((-1, self.rna_input_size))
            atac_sample = atac_sample.reshape((-1, self.atac_input_size))

            return rna_sample, atac_sample

    def __len__(self):
        return self.sample_num
    

class PrepareDataloader:
    def __init__(self, config:Config) -> None:
        self.config = config
        
        self.joint_profiles = {}
        # load RNA data
        self.joint_profiles['gene_expression'] = read_mtx(self.config.dataset_paths['gene_expression'])
        self.joint_profiles['gene_names'] = np.loadtxt(config.dataset_paths['gene_names'], dtype=str)
        self.joint_profiles['gene_barcodes'] = np.loadtxt(config.dataset_paths['gene_barcodes'], dtype=str)

        # load ATAC data
        self.joint_profiles['atac_expression'] = read_mtx(self.config.dataset_paths['atac_expression'])
        self.joint_profiles['atac_names'] = np.loadtxt(config.dataset_paths['atac_names'], dtype=str)
        self.joint_profiles['atac_barcodes'] = np.loadtxt(config.dataset_paths['atac_barcodes'], dtype=str)        
        
        # 对齐rna 和 atac
        share_barcodes, gene_barcode_index, atac_barcode_index = np.intersect1d(self.joint_profiles['gene_barcodes'], 
                                                                             self.joint_profiles['atac_barcodes'], 
                                                                             return_indices=True)
        self.joint_profiles['gene_expression'] = self.joint_profiles['gene_expression'][gene_barcode_index, :]
        self.joint_profiles['atac_expression'] = self.joint_profiles['atac_expression'][atac_barcode_index, :]
        self.joint_profiles['gene_barcodes'] = self.joint_profiles['gene_barcodes'][gene_barcode_index]
        self.joint_profiles['atac_barcodes'] = self.joint_profiles['atac_barcodes'][atac_barcode_index]

        # convert to csr matrix
        self.joint_profiles['gene_expression'] = csr_matrix(self.joint_profiles['gene_expression'], dtype=np.double)
        self.joint_profiles['atac_expression'] = csr_matrix(self.joint_profiles['atac_expression'], dtype=np.double)
        
        # load rna and atac
        trainset = Dataloader(train=True, 
                              rna_expression_matrix=self.joint_profiles['gene_expression'], 
                              atac_expression_matrix=self.joint_profiles['atac_expression']
                              )
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=
                        config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        self.num_of_cell = len(trainset)
        self.rna_input_size = trainset.rna_input_size
        self.atac_input_size = trainset.atac_input_size

        testset = Dataloader(train=False, 
                              rna_expression_matrix=self.joint_profiles['gene_expression'], 
                              atac_expression_matrix=self.joint_profiles['atac_expression']
                              )
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=
                        config.batch_size, shuffle=True, num_workers=0, pin_memory=True)     

    def getloader(self):
        return self.train_loader, self.test_loader, self.rna_input_size, self.atac_input_size, int(self.num_of_cell/self.config.batch_size)

    def gettestdata(self):
        return self.joint_profiles['gene_expression'], self.joint_profiles['atac_expression']

if __name__ == "__main__":
    config = Config()
    train_loader, test_loader, rna_input_size, atac_input_size, num_of_batch = PrepareDataloader(config).getloader()

    print(num_of_batch)
    for step, (x, y) in enumerate(train_loader):
        print(f"Step: {step}, batch_x's size: {x}, batch_y's size: {y}")
        break
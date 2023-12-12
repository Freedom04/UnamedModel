import torch


class Config:
    def __init__(self) -> None:
        self.dataset = "SNAREseq_cellLineMixture"
        self.use_cuda = True
        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')

        if self.dataset == "SNAREseq_P0_BrainCortex":
            self.dataset_paths = {
                "gene_names": 'data/SNAREseq/P0_BrainCortex_SNAREseq_cDNA.genes.tsv',
                "gene_expression": 'data/SNAREseq/P0_BrainCortex_SNAREseq_cDNA.counts.mtx',
                "gene_barcodes": 'data/SNAREseq/P0_BrainCortex_SNAREseq_cDNA.barcodes.tsv',

                "atac_names": 'data/SNAREseq/P0_BrainCortex_SNAREseq_chromatin.peaks.tsv',
                "atac_expression": 'data/SNAREseq/P0_BrainCortex_SNAREseq_chromatin.counts.mtx',
                "atac_barcodes": 'data/SNAREseq/P0_BrainCortex_SNAREseq_chromatin.barcodes.tsv'
            }
            self.batch_size = 256

        elif self.dataset == "SNAREseq_cellLineMixture":
            self.dataset_paths = {
                "gene_names": 'data/SNAREseq/CellLineMixture/gene_names.txt',
                "gene_expression": 'data/SNAREseq/CellLineMixture/gene_expression.mtx',
                "gene_barcodes": 'data/SNAREseq/CellLineMixture/gene_barcodes.txt',

                "atac_names": 'data/SNAREseq/CellLineMixture/atac_names.txt',
                "atac_expression": 'data/SNAREseq/CellLineMixture/atac_expression.mtx',
                "atac_barcodes": 'data/SNAREseq/CellLineMixture/atac_barcodes.txt'
            }
            self.batch_size = 256
        
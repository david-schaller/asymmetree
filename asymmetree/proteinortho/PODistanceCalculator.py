# -*- coding: utf-8 -*-

from Bio import SeqIO, pairwise2
from Bio.SubsMat import MatrixInfo

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from Bio.Phylo.TreeConstruction import DistanceCalculator


def distance_2seqs(seq1, seq2, matrix=MatrixInfo.blosum62,
                   model='blosum62'):

    aln = pairwise2.align.globaldx(seq1, seq2, matrix,
                                   one_alignment_only=True)
    
    # DistanceCalculator only accepts MultipleSeqAlignment
    msa = MultipleSeqAlignment([SeqRecord(Seq(aln[0][0]), id="seq1"),
                                SeqRecord(Seq(aln[0][1]), id="seq2"),
                                ])
    
    calculator = DistanceCalculator(model)
    dm = calculator.get_distance(msa)
    
    return dm.matrix[1][0]
    

if __name__ == "__main__":
    
    record_dict_C = SeqIO.index("test/C.faa", "fasta")
    C_10 = record_dict_C["C_10"].seq
    
    record_dict_E = SeqIO.index("test/E.faa", "fasta")
    E_10 = record_dict_E["E_10"].seq
    
    distance = distance_2seqs(C_10, E_10)
    print(distance)
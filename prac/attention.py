import torch 
import torch.nn.functional as F
import math


def scale_dot_product_attention(Q, K, V, mask =None):

    d_k = Q.size(-1) # gets the final dimension of the Query 

    # @ is batch matrix multiplication 
    scores = Q @ K.transpose(-2,-1) #.transpose(dim1, dim2) swaps the two specified dimensions.

    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask.bool(), -1e9)
    
    attn_weights = F.softmax(scores, dim= -1)

    output = attn_weights @ V

    return output, attn_weights

def verify():

    print("=" * 60)
    print("VERIFY: scaled dot-product attention")
    print("=" * 60)

    torch.manual_seed(42)

    batch, seq, d_k = 1, 5, 4

    Q = torch.randn(batch, seq, d_k)   
    K = torch.randn(batch, seq, d_k)   
    V = torch.randn(batch, seq, d_k)

    
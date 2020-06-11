
from pyinform import transfer_entropy



xs = [0,1,1,3,1.6,0.1,0,5,0]
ys = [0,0,1,3,1.12412,1,0,0.125124124,0]
a = transfer_entropy(xs, ys, k=1)


b = 0
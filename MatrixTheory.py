#正交匹配法 参考https://zhuanlan.zhihu.com/p/52276805
import torch
import torch.nn as nn
import numpy as np
import torchvision

def reconstruct_greedy(K,M,N):
    x=torch.randn((N,1))
    A=torch.randn((M,N))
    y=A@x
    r=y
    xk=0
    for k in range(K):
        #遍历A的原子a,计算其与r的相关度,即对r的贡献
        #maxn=np.argmax(np.array([abs(A[:,n].t()@r) for n in range(N)]))
        maxn=np.argmax(abs(A.t()@r))
        #计算min|y-Anewx|,计算Anew对y的贡献
        #代码？？？
        if k==0:
            Anew=A[:,maxn].reshape(A.shape[0],1)
        else:
            Anew=torch.cat([Anew,A[:,maxn].reshape(A.shape[0],1)],1)
        xk=Anew.pinverse()@y
        r=y-Anew@xk
    zeropad=nn.ZeroPad2d(padding=(0,0,0,N-xk.shape[0]))
    xk=zeropad(xk)
    print(xk)


if __name__ == '__main__':
    K,M,N=8,50,100
    reconstruct_greedy(K,M,N)

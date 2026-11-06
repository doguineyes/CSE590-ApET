"""Read this project's plain CPU tensor dictionaries as NumPy without Torch.

Restricted pickle globals: only tensor rebuild, known storages and OrderedDict.
Never use for arbitrary objects; shapes/strides/storage lengths are checked.
"""
from collections import OrderedDict
import io
import math
import pickle
import zipfile
import numpy as np


def rebuild(storage, offset, shape, strides, requires_grad, hooks, *extra):
    if requires_grad or hooks or extra or len(shape)!=len(strides):
        raise ValueError('Unsupported tensor metadata')
    if any(type(x)!=int or x<0 for x in (*shape,*strides,offset)):
        raise ValueError('Invalid tensor geometry')
    end=offset+sum((d-1)*s for d,s in zip(shape,strides))+1
    if not shape or min(shape)<1 or end>storage.size or math.prod(shape)>20_000_000:
        raise ValueError('Tensor out of bounds')
    return np.ndarray(shape=shape,dtype=storage.dtype,buffer=storage,offset=offset*storage.itemsize,
                      strides=tuple(s*storage.itemsize for s in strides)).copy()


def read_tensors(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        roots=[n[:-8] for n in z.namelist() if n.endswith('data.pkl')]
        if len(roots)!=1: raise ValueError('Expected one tensor dictionary')
        root=roots[0]
        if z.read(root+'byteorder')!=b'little': raise ValueError('Unsupported byte order')
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if (module,name)==('torch._utils','_rebuild_tensor_v2'):return rebuild
                if (module,name)==('collections','OrderedDict'):return OrderedDict
                dtypes={'HalfStorage':'<f2','FloatStorage':'<f4','DoubleStorage':'<f8','LongStorage':'<i8','BoolStorage':'?'}
                if module=='torch' and name in dtypes:return dtypes[name]
                raise ValueError('Forbidden pickle global: '+module+'.'+name)
            def persistent_load(self,pid):
                if len(pid)!=5 or pid[0]!='storage' or pid[3]!='cpu' or not str(pid[2]).isdigit():
                    raise ValueError('Unsupported storage')
                if type(pid[4])!=int or not 0<pid[4]<=20_000_000:raise ValueError('Storage too large')
                a=np.frombuffer(z.read(root+'data/'+str(pid[2])),dtype=np.dtype(pid[1]))
                if a.size!=pid[4]:raise ValueError('Storage size mismatch')
                return a
        value=Reader(io.BytesIO(z.read(root+'data.pkl'))).load()
        if type(value)!=dict or not all(type(k)==str and isinstance(v,np.ndarray) for k,v in value.items()):
            raise ValueError('Expected plain tensor dictionary')
        return value

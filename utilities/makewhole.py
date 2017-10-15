# python 2.7
import os
import sys
from functools import partial

chunksize = 1024

def joinfiles(directory, filename, chunksize=chunksize):
    print ("restoring:", filename, "from directory:", directory)
    if os.path.exists(directory):
        if os.path.exists(filename):
            os.remove(filename)
        output = open(filename, 'wb')
        chunks = os.listdir(directory)
        chunks.sort()
        for fname in chunks:
            print("fname",fname)
            fpath = os.path.join(directory, fname)
            print("fpath",fpath)
            with open(fpath, 'rb') as fileobj:
                for chunk in iter(partial(fileobj.read, chunksize), ''):
                    if not len(chunk):
                        break
                    output.write(chunk)
                    print(fpath ,"------------",len(chunk))
                    
        output.close()

joinfiles('G:\\CarND\\ObjectDetection\\Data_prep\\Images_site\\Fzmodel\\frozen_model_chunks', 
'G:\\CarND\\ObjectDetection\\Data_prep\\Images_site\\Fzmodel\\Reconstruct\\frozen_inference_graph.pb')
print( "Done!")
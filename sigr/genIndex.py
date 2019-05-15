from __future__ import division
import numpy as np


def genIndex(chanums):

      index = []
      i = 1
      j = i+1

      if (chanums % 2) == 0:
         Ns = chanums+1
      else:
         Ns = chanums


      index.append(1)
      t = chr(i+ord('A'))
      while(i!=j):
          l = ""
          l = l+chr(i+ord('A'))
          l = l+chr(j+ord('A'))
          r = ""
          r = r+chr(j+ord('A'))
          r = r+chr(i+ord('A'))
          if(j>Ns):
              j = 1
          elif(t.find(l)==-1 and t.find(r)==-1):
              index.append(j)
              t = t+chr(j+ord('A'))
              i = j
              j = i+1
          else:
              j = j+1



      new_index = []
      if (chanums % 2) == 0:
          for i in range(len(index)):
              if index[i] != chanums+1:
                 new_index.append(index[i])
          index = new_index

      index = np.array(index)
      index = index-1
      return index


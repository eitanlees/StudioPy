"""
Author: Ezra S. Brooker
Date Created: 2021 July 02
Date Modified: 

Dept of Scientific Computing
Florida State University

Proof-of-Concept for basic FFT examples using
the PySimpleGUI package for generating the GUI

Recursive FFT/DFT

"""
import sys
if sys.version_info[0] < 3:
  raise Exception("Python 2 is no longer supported, please use Python 3!")

import os
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from collections import OrderedDict
import numpy as np

# Might need to install
import PySimpleGUI as sg
from graphviz import Digraph, Source
# import pydot

from iea.utils.base_window import BaseWindow


def algorithm_text(line=[24]):
  # Dirty way of dealing with the changing algorithm text format as we cycle through portions of it
  # The function is called by window["-MULTI-"].update(algorithm_text()) to update the text source
  # And window["-MULTI-"].update() is called most by self._wait_for_update_call() that allows us to control
  # our stepping through the recursive algorithm
  prestr = ["    " for _ in range(25)]
  for i in line:
    prestr[i] = "--->"
  string  = "                                                                             \n\n"
  string += "       <--- Recursive FFT Algorithm --->                                       \n"
  string += "                                                                               \n"
  string += " 0  {0}    procedure fft( x )                                                  \n".format(prestr[0] )
  string += " 1                                                                             \n"
  string += " 2  {0}        n = size(x)                                                     \n".format(prestr[2] )
  string += " 3  {0}        if n = 1 then                                                   \n".format(prestr[3] )
  string += " 4  {0}            return x[ 0 ]                                               \n".format(prestr[4] )
  string += " 5                                                                             \n"
  string += " 6  {0}        else                                                            \n".format(prestr[6] )
  string += " 7  {0}            even = x[ ::2  ]                                            \n".format(prestr[7] )
  string += " 8  {0}            odd = x[ 1::2 ]                                             \n".format(prestr[8] )
  string += " 9                                                                             \n"
  string += "10  {0}            x = fft( even )                   # FFT over even indices   \n".format(prestr[10])
  string += "11  {0}            x += fft( odd  )                  # FFT over odd  indices   \n".format(prestr[11])
  string += "12                                                                             \n"
  string += "13  {0}            T = exp( ( -2 * pi * 1j ) / n )   # Twiddle factor          \n".format(prestr[13])
  string += "14  {0}            for k1 = n // 2                                             \n".format(prestr[14])
  string += "15  {0}                xk = x[ k1 ]                                            \n".format(prestr[15])
  string += "16  {0}                k2 = k1 + n // 2                                        \n".format(prestr[16])
  string += "17  {0}                x[ k1 ] = xk + x[ k2 ] * T^k1                           \n".format(prestr[17])
  string += "18  {0}                x[ k2 ] = xk - x[ k2 ] * T^k1                           \n".format(prestr[18])
  string += "19  {0}            end                                                         \n".format(prestr[19])
  string += "20  {0}        end                                                             \n".format(prestr[20])
  string += "21                                                                             \n"
  string += "22  {0}        return x                                                        \n".format(prestr[22])
  string += "23  {0}    end procedure                                                       \n".format(prestr[23])
  string += "                                                                             \n\n"
  return string


# Things that controls our FFT directed graph objects and connections
class FFT_graphable_num:

  node = 0
  all = []
  max_level = 0
  max_index = 0
  all_by_level = OrderedDict()

  @classmethod
  def clear(cls):
    cls.node = 0
    cls.all = []
    cls.max_level = 0
    cls.max_index = 0
    cls.all_by_level = OrderedDict()


  def __init__(self, number, level, parents = None, index = 0, index_offset = 1):

    if isinstance(number, FFT_graphable_num):
      number = number.complx
    self.complx = number
    self.real = self.complx.real
    self.imag = self.complx.imag
    self.string = str(round(self.real,2)) + " + " + str(round(self.imag,2)) + "i"
    self.level = level                                                          #this is the y index in the graphviz graph
    self.parents = parents
    FFT_graphable_num.max_level = max(level, FFT_graphable_num.max_level)

    FFT_graphable_num.node += 1
    self.node = str(FFT_graphable_num.node)

    self.set_index(index, index_offset)                                         #this sets the x index in the graphviz graph

    FFT_graphable_num.all.append(self)
    FFT_graphable_num.all_by_level[level].append(self)
    FFT_graphable_num.all_by_level[level].sort(key=lambda x: x.index)


  def __add__(self, other):
    if isinstance(other,FFT_graphable_num):
      other = other.complx
    return self.complx + other


  def __sub__(self, other):
    if isinstance(other,FFT_graphable_num):
      other = other.complx
    return self.complx - other


  def __mul__(self, other):
    if isinstance(other,FFT_graphable_num):
      other = other.complx
    return self.complx * other


  def set_index(self, index = 0, offset = 0):
    """
    set horizontal index of node given other nodes in graph
    """
    if self.level not in FFT_graphable_num.all_by_level:
      FFT_graphable_num.all_by_level[self.level] = []
      self.index = 0                                                            #if this level hasn"t been seen yet, we"re on the far left side
      return

    indices_on_my_level = [node.index for node in
                                    FFT_graphable_num.all_by_level[self.level]]
    while index in indices_on_my_level:
      index += offset
    self.index = index
    FFT_graphable_num.max_index = max(self.index, FFT_graphable_num.max_index)


class SubModuleWindow(BaseWindow):

  title = "FFTs: Recursive DFT"
  _fout = "temp_recurse_fft_graph"
  wincfg = {}
  _counter = 0
  _imgs = []
  _algo = []

  def __init__(self, configfile=None, launch=False):
    super().__init__(configfile, launch)


  def _configure_layout(self):
    sg.theme("Dark") # window theme

    # create a column in the window for the input boxes
    col_input = [
      [ sg.Text("", size=(1,1))                                                                                              ],
      [ sg.Text("FFT not initialized...", size=(55,1),key="-PROGRESS-")                                                       ],
      [ sg.Button("Submit", bind_return_key=True, size=(5,1)), sg.InputText("1+3j, 2, 3, 4+1j", key="-INPUT-" )              ],
      [ sg.Button("Reset",  bind_return_key=True, size=(5,1)), sg.Text("Reset Graph: Click or press  <r>  key", size=(50,1)) ],
      [ sg.Button("Next",   bind_return_key=True, size=(5,1)), sg.Text("Advance FFT: Click or press  <w>  key", size=(50,1)) ],
      [ sg.Button("Exit",   bind_return_key=True, size=(5,1)), sg.Text("Exit Window: Click or press <ESC> key", size=(50,1)) ],
      [ sg.Text("", size=(1,1))                                                                                              ],
    ]

    # We"ll place the algorithm text in the scrollable box below the input line and buttons
    col_multi = [
      [ sg.MLine(algorithm_text(), size=(70,35), key="-MULTI-") ]
    ]

    # create a column in the window for displaying the Canvas plot
    col_image = [ 
      [sg.Image(filename=self._fout, key="-IMAGE-", enable_events=True)],
    ]


    # assign columns to layout list in order they should appear
    self.layout = [
      [
        sg.Pane([sg.Col(col_input),sg.Col(col_multi)]),
        sg.VSeperator(), # vertical line seperator between columns
        sg.Col(col_image)
      ]
    ]

  def launch(self):

      self.wincfg["finalize"]  = True
      self.wincfg["return_keyboard_events"] = True
      self.wincfg["resizable"] = True
      self.wincfg["location"]  = [100,100]
      self._default_graph(update_window=False) # Make a first call to generate it window startup
      super().launch()
      self.window["-MULTI-"].update(algorithm_text(line=[0]))
      self.inp = "0+0j, 0, 0, 0+0j"
      self._counter = 0
      self._imgs = []
      self._algo = []


  def check_read(self,event,values):

    if event in (None, "Exit", sg.WIN_CLOSED, "\x1b"):
      self.window.close()
      return True

    elif event in ("Submit", "s"):
      self._counter = 0
      self._imgs = []
      self._algo = []
      # Clear any attributes from FFT graph
      FFT_graphable_num.clear()
      self.inp = values["-INPUT-"]
      x = [complex(i.replace(" ","")) for i in self.inp.split(",")]
      x = self._fft(x)
      self.window["-PROGRESS-"].update("FFT initialized! You may proceed...")


    elif event in ("Next", "w") and self._counter < len(self._imgs):
      self._imgs[self._counter].write(self._fout,format="png")
      self.window["-IMAGE-"].update(filename=self._fout)
      self.window["-MULTI-"].update(self._algo[self._counter])
      self._counter+=1

      if 0 != self._counter >= len(self._imgs):
        self.window["-PROGRESS-"].update("FFT procedure finished, RESET or SUBMIT to continue")

    elif event in ("Reset", "r"):
      # User gave reset graph command
      self._default_graph(data=self.inp, update_window=True)
      self.window["-MULTI-"].update(algorithm_text(line=[0]))
      self.window["-PROGRESS-"].update("FFT not initialized...")
      self._counter = 0
      self._imgs = []
      self._algo = []

    return False

  # The directed graph is generated here using the FFT_graphable_num class and Graphviz
  def _update_graph(self,update_window=False): 
    dot = Digraph(format="png")
    dot.attr("node", shape="rectangle")
    for num in FFT_graphable_num.all:
      dot.node(num.node, f"{round(num.real,2)} + {round(num.imag,2)}i")
    for i in range(FFT_graphable_num.max_level+1):                                #add invisible edges to side-by-side nodes to make graph look nice
      with dot.subgraph() as s:
        s.attr(rank="same")
        pre_node = FFT_graphable_num.all_by_level[i][0].node
        level_indices = [node.index for node in FFT_graphable_num.all_by_level[i]]
        for j in range(1, FFT_graphable_num.max_index+1):
          if j in level_indices:
            post_node = list(filter(lambda x: x.index == j, 
                          FFT_graphable_num.all_by_level[i]))[0].node
          else:
            s.node("_" + str(i)+str(j), f"{1.0} + {1.0}i", style="invis")         #make a ghost node if needed
            post_node = "_" + str(i) + str(j)
          s.edge(pre_node, post_node, style="invis")
          pre_node = post_node
    for num in FFT_graphable_num.all:                                             
      if num.parents is not None:
        has_parent_above = False                                                  
        for p in num.parents:
          dot.edge(p.node, num.node)
          if p.index == num.index:
            has_parent_above = True
        if not has_parent_above:                                                  #create an invisible, vertical edge to make the graph look nice
          potential_parent = [x for x in                                          #check if there"s a real parent above the node
              FFT_graphable_num.all_by_level[num.level-1] if x.index == num.index]
          if potential_parent:
            dot.edge(potential_parent[0].node, num.node, style="invis")
          else:                                                                   #if not, align with the "ghost" node (created above)
            dot.edge("_" + str(num.level-1)+str(num.index), num.node, style="invis")

    gr = pydot.graph_from_dot_data(dot.source)[0]  # Convert to pydot Digraph
    gr.write(self._fout,format="png")         # Write to temporary file name

    # Update graph window if desired    
    if update_window:
      self.window["-IMAGE-"].update(filename=self._fout)
    else:
      self._imgs.append(gr)

    

  # Actual recursive FFT function we use (it"s a little different than Heath"s)
  def _fft(self,x, parents=None):

    N = len(x)
    assert np.log2(N) % 1 == 0   #our code assumes we have an array length that"s a power of 2
    new_x = x.copy()

    # Generate the object and attributes for the directed graph
    for i in range(len(x)):
      if parents is None:
        new_x[i] = FFT_graphable_num(x[i], level = 0)
      else:
        new_x[i] = FFT_graphable_num(x[i], level = parents[i].level + 1,
                                                parents = [parents[i]])

    
    if parents == None:
      # Update graph window immediately, if first (Base level==0) call
      self.window["-MULTI-"].update(algorithm_text(line=[0]))
      self._update_graph(True)
    elif parents!=None and N > 1:
      # If child FFT call, then wait for user update command
      self._wait_for_update_call(position=[0])

    x = new_x
    
    if N > 1:

      # Gather your even and odd array elements together
      self._wait_for_update_call(position=[7,8])
      even = x[::2]
      odd = x[1::2]

      # Perform FFT on even elements (each call cuts the sequence down by half)
      self._wait_for_update_call(position=[10])
      x = self._fft(even, parents = even)

      # Perform FFT on odd elements (each call cuts the sequence down by half)
      self._wait_for_update_call(position=[11])
      x += self._fft(odd, parents = odd)

      # Get your twiddle factor for the final calculations
      self._wait_for_update_call(position=[13])
      T = np.exp(-2*np.pi*1j/N)

      self._wait_for_update_call(position=[14])
      for k in range(N//2):
        # Final calculation with Twiddle factor power scaling
        parents = x.copy()
        xk = x[k]

        x[k] = FFT_graphable_num(xk + x[k+N//2] * T**k, 
                                 level = parents[k].level + 1, 
                                 parents = [parents[k], parents[k+N//2]],
                                 index = k,
                                 index_offset = N//2)

        x[k+N//2] = FFT_graphable_num(xk - x[k+N//2] * T**k, 
                                      level = parents[k].level + 1, 
                                      parents = [parents[k], parents[k+N//2]],
                                      index = k + N//2,
                                      index_offset = N//2) 

      self._wait_for_update_call(position=[22])
      # Successful calculation of this FFT call, we can jump up one level

    else:
      # Only one array element on this call, so we just say x[0] = x[0]
      # and jump out of this level of the recursion, successful return

      self._wait_for_update_call(position=[4])

    return x


  def _default_graph(self,data="0+0j, 0, 0, 0+0j", update_window=False):
    # Default graph with all zeros
    if update_window: self.window["-MULTI-"].update(algorithm_text())
    FFT_graphable_num.clear()
    x0 = [complex(i.replace(" ","")) for i in data.split(",")]
    for i in range(len(x0)):
      x0[i] = FFT_graphable_num(x0[i], level = 0)
    self._update_graph(update_window=update_window)

  
  def _wait_for_update_call(self,position=[24], regraph=True):

      self._algo.append(algorithm_text(line=position))
      # self.window["-MULTI-"].update(algorithm_text(line=position))
      if regraph:
        self._update_graph()
      return


if __name__ == "__main__":

  pass
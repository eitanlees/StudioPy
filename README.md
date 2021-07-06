# DSC_IEA
Interactive Educational Applet for the Florida State University Department of Scientific Computing courses. Inspired by the interactive educational modules developed for Michael Heath's scientific computing textbook.

# Use
At this time, the interactive educational applet is best installed through `conda` to easily obtain the necessary package `python-graphviz` due to an annoying issue found on Windows 8 (not tested on Windows 10, yet). There are ways to get around this by installing Graphviz and dealing with PATH names appropriately, but using `conda` is simply easier.

Suggested setup is via a fresh `conda` environment:

`conda create --name ENV-NAME -c conda-forge --file requirements.txt`

`conda activate ENV-NAME` or `source activate ENV-NAME` if using MacOS

`python main.py` in the parent level DSC_IEA directory.
    
Bug Notes: There are some small bugs littered throughout that can cause crashes from unexpected user input. Those will be solved with some exception handling.


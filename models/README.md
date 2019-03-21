Requirements
------------

This project is coded in Python 3.6. Required packages include Numpy, Pandas and Matplotlib. Executes

>pip3 install numpy matplotlib pandas

or

>pip3 install -r requirements.txt

Generate Figures Using Random Training Sets
------------

Execute the following line to generate all three figures using new randomly generated training sets. Numerical results will also be displayed in the terminal.

>python3 all_figures.py

Then generated figures will be in the folder `figures` folder. The file name of the reproduced figures will look like `fig3_1550443283.png` where `1550443283` is a time stamp, so that new generated figures will not overwrite old ones.

To generate each figure separately, run the following commands,

>python3 figure3.py

>python3 figure4.py

>python3 figure5.py

For reproducing `Figure 4`, the error TD(0) at α=0.6 could be between TD(0.3) and TD(0.8). Run the command `python3 figure4.py` three times or more to see a plot where the error TD(0) is above the error of TD(0.8), as shown in my report.

Generate The Same Figures in The Report
------------

Execute the following line to generate exactly the same figures in my report, using files in the `data` folder. (However, this might fail due to compatibility problem of different Numpy and Pickle versions.)

>python3 all_figures.py -f

>python3 figure3.py -f

>python3 figure4.py -f

>python3 figure5.py -f

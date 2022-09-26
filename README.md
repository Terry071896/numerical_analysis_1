# numerical_analysis_1
This is for Introduction to Numerical Analysis I at University of Utah MATH 6860


## Installing

```{bash}
git clone https://github.com/Terry071896/numerical_analysis_1.git
cd numerical_analysis_1
```

From here with conda:

```{bash}
conda env create -f environment.yml
conda activate numericalI
```

or with pip:

```{bash}
pip install -r requirments.txt
```

## Info

Each homework will have a pdf of the full homework, a notebook file which is recommended to to run code, and a file that is the notebook condensed as a file (won't run as nicely as the notebook)

If you are having issues running the notebook after you have created the virtual environment, then do this step:
```{bash}
conda activate numericalI
conda install ipykernel
ipython kernel install --user --name="numericalI"
cd /foo/numericalI
jupyter lab
```
Now in jupyter lab find the notebook to run and change the kernel to "numericalI"

This all should be workable in VSCode as well.

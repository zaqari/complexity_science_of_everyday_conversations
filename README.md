# A set of dynamical rules for conversational progression act as a precursor to collective cognition

How is it that people work together to progress a conversation? If you started from any random utterance in a conversation, how would each speaker and their partners create and compound information? And how does that potentially support distributed computation, and collective cognition?

The objective of this project is to assess these questions. We analyze 5,448 conversations to get at answers to this problem.

> [A webpage containing complete study information, methods descriptions (including an interactive visualization of method steps), and interactive results visualizations.](https://zaqari.github.io/complexity_science_of_everyday_conversations/web/report.html)


## Process/Steps
The scripts contained here ought to be run in the following steps:

You can circumvent having to run steps in both (1) and (2) by [downloading the data from the following UCLA box folder](https://ucla.box.com/s/8qhlrqvh7i2g3vtzbcrmc17y8swr3td8).

### (1) Data pre-processing
1. Collect data from the sources required
   1. Download .cha files from talkbank manually, then place them in the folder `data/chas/`
   2. request and download candor files, then place them in the folder `data/candor/`
2. Run the scripts contained in preprocessing labeled `0-`

### (2) Calculating information for pairwise comparisons
In an earlier version, we needed to ascertain how much using entropy versus just raw information caused measurement variances. For this reason, we ended up taking a multi-step process to calculate both of these measurements. The scripts for this are contained in the directory `server-side-scripts`. Run them in the following order:
1. Run `shape_COS.py` to get strings of maximum CoS values (min CoE) for all comparisons 
2. Run `process_COS_data_METHODSTEST.py`
3. scp files from server to local computer

### (3) Calculating CDF and downsampling data
Once you have the data from the server, the following will calculate the CDF values used in the study:
1. Use `calculate_CDF.ipynb` to calculate CDFs for each utterance. 
   - The notebook will also downsample the data to be in an exponential distribution. You can easily comment out the portions doing this to remove them. Downsampling simply makes final analyses a little faster.
   - This handles both observed and null permuted data.

### (4) Analyses
The final analyses are split according to english monolingual study and crosslignuistic comparison study. The notebooks for those studies are contained in
   - English monolingual: `distCog/eng/main_process`
   - Crosslinguistic comparison: `distCog/xling/main_process`
Run all the notebooks in both.

### (5) study visualizations
Visualizations are created using the notebooks contained in `distCog/vis`.
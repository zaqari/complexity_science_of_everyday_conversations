# Understanding Universal Conceptual Organizations in Conversational Dynamics

The basic premise is this: there appears to be two underlying universal, dynamical structures underlying conversational turns and their lexico-semantic content:

1. Turns that are closer to one another in time have lower convergence-entropy in comparison to one another (likely because turns in close temporal proximity share conceptual content by dint of people needing to be somewhat aligned in their pragmatic understanding of the world in order to maintain a conversation)
2. People tend to converge more deeply with the conceptual content of their own turns than they do with others.

This project is designed to, through analysis of a truly ridiculous number of conversations, assess these claims.

## Process/Steps
The scripts contained here ought to be run in the following steps:

1. Collect data from the sources required
   1. Download .cha files from talkbank manually, then place them in the folder `data/chas/`
   2. request and download candor files, then place them in the folder `data/candor/`
2. Run the scripts contained in preprocessing labeled `0-`
3. Run `shape_CEDA.py`, designating whether one is working with the cha file corpus or the candor corpora.
4. Use `2-CIT-Aanalysis.ipynb` to complete data analysis.
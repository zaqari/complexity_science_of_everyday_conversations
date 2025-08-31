# The Complexity Science of Everyday Conversation: Selves, Others, and Drift between Concepts in Dialog

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
3. Run `shape_CEDA.py` on a server containing the data you're looking for.
4. Use `2-CIT-Aanalysis.ipynb` to complete data analysis.

## Process from the perspective of the itkin server
- [ ] Create aligned documents using files in `0-preprocessing`
- [ ] scp /Volumes/TOREN/comp_ling/datasci/ComplexityScienceOfEverydayConvos/english/data/server_ready/null-corpus.parquet zprosen@itkin.comm.ucla.edu:/home/zprosen/d/shapeoflang/raw
- [ ] python3 ./shape_CEDA.py
- [ ] scp zprosen@itkin.comm.ucla.edu:/home/zprosen/d/shapeoflang/ckpts/graph-obj-null-corpus.parquet.pt /Volumes/TOREN/comp_ling/datasci/ComplexityScienceOfEverydayConvos/english/data/ckpts 
- [ ] Stich document (1-Stitching-ckpts-CHAs.ipynb)
- [ ] Create omni-document (2-Create-Omni-File.ipynb)
- [ ] Calculate residual (3.1… .ipynb)
- [ ] Null test (3.2-Null-Hyp.ipynb)
- [ ] Calculate Allan variance (4.allan_variance.1 .ipynb)
- [ ] scp /Volumes/TOREN/comp_ling/datasci/ComplexityScienceOfEverydayConvos/english/4-complexity_tests/allan-variance/allan_var.parquet [zprosen@](mailto:zprosen@itkin.comm.ucla.edu)itkin.comm.ucla.edu:/home/zprosen/d/allan_variance/
- [ ] Update allan_factor_server.py
- [ ] scp zprosen@itkin.comm.ucla.edu:/home/zprosen/d/allan_variance/av-params.csv /Volumes/TOREN/comp_ling/datasci/ComplexityScienceOfEverydayConvos/english/4-complexity_tests/allan-variance/
- [ ] Power-law analysis (4.allan_variance.2 .ipynb)

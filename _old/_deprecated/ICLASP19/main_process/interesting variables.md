# A list of interesting CANDOR Variables to analyze

There are a lot of possible variables to analyze in CANDOR and all of them are interesting. I'm running separate notebooks for each variable partially for ease of replication. But this is the entire list of variables I'm interested in.

The main hypothesis is that a given slope for gradient, log-transformed shifts in conceptual structures is  conditioned on survey response variables (in the list) and whether slopes are for self-versus-other.

$$\begin{align}s_i &\sim \mathcal{N} \bigg(\beta_{c} + \psi_i\ v_i, \ \sigma^2_{p} \bigg) \\ \psi_i &= \gamma \ \delta_{i\neq self} + \phi\ \delta_{i=self} \\ \beta_c &= \beta_{self} \ \delta_{i=self} + \beta_{i \neq self} \ \delta_{i \neq self} \\ \delta_c &= \begin{cases} 1 & \text{iff}\ \ c \\ 0 & \text{elsewhere} \end{cases} \end{align}$$

[More details about the variables can be found here](https://docs.google.com/spreadsheets/d/1ADoaajRsw63WpM3zS2xyGC1YS5WM_IuhFZ94W84DDls/edit?gid=997152539#gid=997152539)

Current best results and model version is stored in `./_older/20251209-continuous-betas/`

## Merge docs
Merge all the docs from 202603031630 on to make one doc with multiple chains.

## All variables of interest

### Affective motives
- [x] best_affect

- [x] conv_leader

- [x] conversationalist

- [x] how_enjoyable

- [x] in_common
    
- [x] i_like_you

- [x] i_felt_close_to_my_partner

- [x] i_think_your_status

- [x] i_would_like_to_become_friends

- [x] overall_affect

- [ ] responsive_1
    
- [ ] responsive_2
    
- [ ] responsive_3

- [x] you_are_friendly

- [ ] you_are_good_listener

- [x] you_are_intelligent

- [ ] you_are_kind
    
- [x] you_are_quickwitted

- [ ] you_are_warm

- [ ] you_like_me

### Cognitive motives
- [x] anticipated_each_other_sr6

- [x] developed_joint_perspective_sr2
    
- [x] our_thoughts_synced_up_sr1

- [x] saw_world_in_same_way_sr8

- [x] shared_reality

- [x] thoughts_became_more_alike_sr5
    

    

    

# Dobble symbols dataset

2022-03-21 J. Chazalon with help from D. Godin

Dobble symbols are copyrighted by Denis Blanchot.


The dataset is split into two sets:

- a train set
- a test set


File organization for the train set:

- one directory per symbol
- one image per sample (the same is `c${XX}_s${YY}.png` where `${XX}` is our internal card id and `${YY}` is our internal symbol id on the card).


File organization for the test set:

- no directory structure
- the `gt.txt` file contains the mapping `class_id` ↔ `filename`


The test set will be made available to students only during the defense, for the live demo.



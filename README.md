# binary-isa-classifier

This project was initiated primarily to solve Praetorian's challenge of predicting ISA given binary data.  I plan to extend the classifier outside the scope of the challenge moving forward.

The model is trained to classify the following list of architectures:
- x86_64
- arm
- avr
- alphaev56
- m68k
- mipsel
- mips
- powerpc
- s390
- sh4
- sparc
- xtensa

### repo description

- ``main.py``: interacts with Praetorian's server to collect data and test model performance.
- ``modeling.ipynb``: for analysis, experiments, and trying different models.
- ``ensemble_train.ipynb``: script to experiment with an ensemble model.  very cursory atm.
- ``data``: self-explanatory directory.  use ``bin-data.csv`` for testing model viability.
- ``models``: serialized models.

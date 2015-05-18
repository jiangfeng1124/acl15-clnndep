##Cross-lingual Dependency Parsing Based on Distributed Representations
Jiang Guo, jiangfeng1124@gmail.com

###Compile

Require: cmake (available in most Linux systems)

* ./configure
* make

###Resources

The cross-lingual word embeddings generated using PROJ/CCA can be obtained from [here](https://drive.google.com/file/d/0B1z04ix6jD_Db3REdHlnREpjMmc/view?usp=sharing).
Note that the <b>EN</b> word embeddings in PROJ are used only for initialization, they get updated/finetuned while training.

###Running the executable

For robust projection: `./train-PROJ.sh`
* set `fix_word_embeddings=false` (```conf/nndep.cfg```)

For CCA: `./train-CCA.sh`

Learning parameters are defined in ```conf/nndep.cfg```.


##Cross-lingual Dependency Parsing Based on Distributed Representations
Jiang Guo, jiangfeng1124@gmail.com

###Compile

Require: cmake (available in most Linux systems)

* ./configure
* make

###Resources

Alignment dictionaries:
* ```resources/align/PROJ``` (target->source)
* ```resources/align/CCA``` (source->target).

Note that the <b>EN</b> word embeddings in PROJ are used only for initialization, they get updated/finetuned while training.

###Running the executable

For robust projection: `./train-PROJ.sh`
* set `fix_word_embeddings=false` (```conf/nndep.cfg```)

For CCA: `./train-CCA.sh`

Learning parameters are defined in ```conf/nndep.cfg```.


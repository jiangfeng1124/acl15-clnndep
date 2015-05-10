##Cross-lingual Dependency Parsing Based on Distributed Representations
Jiang Guo, jiangfeng1124@gmail.com

###Compile

Require: cmake (available in most Linux systems)

* ./configure
* make

###Resources

The cross-lingual word embeddings generated using PROJ/CCA can be downloaded from [here](https://drive.google.com/file/d/0B1z04ix6jD_Db3REdHlnREpjMmc/view?usp=sharing)

###Running the executable

For robust projection: ```./train-PROJ.sh```

For CCA: ```./train-CCA.sh```

Learning parameters are defined in ```conf/nndep.cfg```.


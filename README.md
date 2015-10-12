##Cross-lingual Dependency Parsing Based on Distributed Representations
Jiang Guo, jiangfeng1124@gmail.com

This parser can be used both for monolingual and cross-lingual dependency parsing.
The monolingual component has been optimized and integrated into **[LTP](https://github.com/HIT-SCIR/ltp)**.

###Compile

Require: cmake (available in most Linux systems)

* ./configure
* make

###Resources

Alignment dictionaries:
* ```resources/align/PROJ``` (target->source)
* ```resources/align/CCA``` (source->target).

Unzip the embeddings:
* ```cd resources && unzip acl15-cl-wemb.zip && mv acl15-cl-wemb/cca cca && mv acl15-cl-wemb/projected projected && rm -r acl15-cl-wemb && cd ..```

Note that the <b>EN</b> word embeddings in PROJ are used only for initialization, they get updated/finetuned while training.

###Running the executable

For robust projection: `./scripts/train-PROJ.sh`
* set `fix_word_embeddings=false` (```conf/nndep.cfg```)

For CCA: `./scripts/train-CCA.sh`

Learning parameters are defined in ```conf/nndep.cfg```.

###Reference

```
@InProceedings{Guo:2015:clnndep,
  author    = {Guo, Jiang and Che, Wanxiang and Yarowsky, David and Wang, Haifeng and Liu, Ting},
  title     = {Cross-lingual Dependency Parsing Based on Distributed Representations},
  booktitle = {Proceedings of ACL},
  year      = {2015},
}
```


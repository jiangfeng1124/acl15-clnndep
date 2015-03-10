Current projection process:

0. python project_cls.py [en_cluster] [alignment] [target_lang_projected_cluster]

    # Example:
        python project_cls.py en.256 align/de-en.align de.256

1. python post_process_de.py [depfile] [en_cluster] [fo_cluster] > [OOV_clusters]

    # Example:
        python post_process_de.py /udt/de/de-universal-test.conll en.256 de.256 > de.oov

2. python [fo_cluster] [oov_cluster] > [fo_final_cluster]

    # Example:
        cat de.256 de.oov > de.256.p

Step 1/2 should be merged.


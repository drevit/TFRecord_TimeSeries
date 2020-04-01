BUFFER_SIZE    = 8192       # buffer di dati paricati in prefetch e shufflati
SEED           = 42         # seed di np.random.seed, per riproducibilità risultati
SEQUENCE_LENGTH = 100        # lunghezza della sequenza della RNN
BATCH_SIZE     = 512        # default 512. Almeno 128 e possibilmente una potenza di 2
DATASET_FORMAT = 'tfrecord' # 'tfrecord' o 'numpy'. formato di salvataggio dati
DEBUG_RNN = True

save_dir = '/data'

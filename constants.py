SEQ_LENGTH = 384
BATCH_SIZE = 16
EPOCHS = 4  # for fine-tuning, especially transformers, 3-5 is fine
MAX_LR = 1e-4
N_BEST = 10
MAX_ANSWER_LEN = 30
NUM_EPOCH_WARMUP = 0.5  # LR warmup length
COMPUTE_PER_EPOCH = (
    10  # approx. number of train and val loss computations/prints per epoch
)

PATH = "./"

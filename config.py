from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf

class Config:
    """
    Configuration for the model and the training process
    """
    def __init__(self):
        # Training parameters
        self.epochs = 10  # Number of training epochs
        self.batch_size = 64  # Batch size
        self.lr = 0.001  # Learning rate
        self.weight_decay = 0.0001  # Weight decay
        self.save_dir = './checkpoints'  # Path to save the model
        self.loss_weights = [1.0, 1.0, 1.0]  # Weights for the three losses
        self.batch_size_test = 100  # Batch size for fine tune
        self.lrate_decay = 0.95  # Decay of learning rate every 100 iterations
        self.num_iterations_per_decay = 100
        self.gama = 1e-6 # contribution of imbalance loss

        # Model parameters
        self.input_dim = 100  # Input dimension
        self.hidden_dim = 256  # Hidden layer dimension
        self.optimizer = Adam(learning_rate=self.lr, decay=self.weight_decay)

        self.epsilon = 1e-3
        self.dropoutrate = 0.1
        self.num_layer = 3      # Encoder & Decoder层数
        self.num_heads = 5      # Attention头数


        # Added parameters
        self.n_in = 2  # Number of representation layers
        self.n_out = 2  # Number of regression layers
        self.p_alpha = 1e-8  # Imbalance regularization param
        self.p_lambda = 1e-6  # Weight decay regularization parameter
        self.dropout_in = 0.9  # Input layers dropout keep rate
        self.dropout_out = 0.9  # Output layers dropout keep rate
        self.dim_in = 100  # Pre-representation layer dimensions
        self.dim_out = 100  # Post-representation layer dimensions
        self.batch_norm = 1  # Whether to use batch normalization
        self.normalization = 'none'  # How to normalize representation
        self.rbf_sigma = 0.1  # RBF MMD sigma
        self.experiments = 1  # Number of experiments
        self.iterations = 1000  # Number of iterations training
        self.iterations_tune = 500  # Number of iterations testing
        self.weight_init = 0.01  # Weight initialization scale
        self.optimizer_name = 'Adam'  # Which optimizer to use

config = Config()
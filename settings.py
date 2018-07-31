"""
The settings for a run.
"""
class Settings:
    def __init__(self):
        self.trial_name = 'cnn'
        self.log_directory = '/media/pankaj/D04BA26478DAA96B/PANKAJ/logs'
       # self.train_dataset_path = '/media/pankaj/D04BA26478DAA96B/PANKAJ/noleaky'
        self.validation_dataset_path = '/media/pankaj/D04BA26478DAA96B/PANKAJ/testdatabase/'
        self.test_dataset_path = self.validation_dataset_path
        self.load_model_path = '/media/pankaj/D04BA26478DAA96B/PANKAJ/jointcnn/CNN 5 Cameras 5 Images lr 1e-5 y2018m05d04h11m41s21/model 1000000'
        self.summary_step_period = 1000
        self.number_of_epochs = 100000
        self.batch_size = 100
        self.number_of_data_loader_workers = 0
        self.save_epoch_period = 10000
        self.restore_mode = 'transfer'
        self.loss_order = 1
        self.weight_decay = 0.01

        self.unlabeled_loss_multiplier = 1e-3
        self.fake_loss_multiplier = 1e-6
        self.mean_offset = 0
        self.learning_rate = 1e-3
        self.gradient_penalty_on = False
        self.gradient_penalty_multiplier = 1
        

        self.initial_learning_rate = 1e-3
        self.learning_rate_multiplier_function = lambda epoch: 0.1 ** (epoch / 2000)

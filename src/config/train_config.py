class OCRTrainConfig:
    synthetic_batch_size = 32
    natural_batch_size = 16

    synthentic_lr = 5e-5
    natural_lr = 5e-5

    synthetic_decay_rate = 0.01
    natural_decay_rate = 0.01

    synthetic_epochs = 50
    natural_epochs = 50

    synthetic_checkpoint_path = "~/synthetic-resnet-50.pt"
    natural_checkpoint_path = "~/synthetic-natural-resnet-50.pt"

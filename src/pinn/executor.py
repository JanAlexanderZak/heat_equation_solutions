""" Executable example of 1D heat equation in PyTorch Lightning.
"""
import pytorch_lightning as pl
import torch

from src.pinn.dl import DeepLearningArguments
from src.pinn.model import HeatEq1DPINNRegressor
from src.pinn.data_module import HeatEq1DPINNDataModule


def main():
    args = DeepLearningArguments(
        seed=6020,
        batch_size=50,
        max_epochs=50000,
        min_epochs=1000,
        num_workers=6,
        accelerator="cpu",
        devices=-1,
        sample_size=1,
        pin_memory=True,
        persistent_workers=True,
    )

    hyper_parameters = {
        "activation_function": torch.nn.Tanh,
        "layer_initialization": torch.nn.init.xavier_uniform_, #torch.nn.init.xavier_normal_
        "optimizer": torch.optim.Adam,
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "scheduler_patience": 100,
        "weight_decay": 1e-3,
        "scheduler_monitor": "train_loss",
        "learning_rate": 1e-4,
        "loss_IC_param": 10,
        "loss_BC_param": 1,
        "loss_PDE_param": 1,
        "num_hidden_layers": 3,
        "size_hidden_layers": 300,
        "batch_normalization": False,
    }

    data_module = HeatEq1DPINNDataModule(
        path_to_data="./src/pinn/data/",
        args=args,
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    #val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    model = HeatEq1DPINNRegressor(
        hyper_parameters=hyper_parameters,
        in_features=data_module.in_features,
        out_features=data_module.out_features,
        column_names=data_module.column_names,
        target_names=data_module.target_names,
    )
    model.hparams.update(data_module.hparams)
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        sync_batchnorm=args.sync_batchnorm,
        min_epochs=args.min_epochs,
    )
    print(dict(model.hparams))

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
    )

    u_pred = trainer.predict(model, dataloaders=test_loader,)       
    torch.save(u_pred, f"./src/pinn/data/predictions/predictions.pkl")


if __name__ == "__main__":
    main()

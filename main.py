from constants import *
from preprocessing import train_dataloader, val_dataloader
from utils import set_seed, device
from train import prep_train, train, plot_losses
from eval import eval_acc

def main():
    set_seed()
    
    model, optimizer, scheduler = prep_train()
    model, train_losses, val_losses = train(model, train_dataloader, val_dataloader,
                                            optimizer, scheduler, device)
    plot_losses(train_losses, val_losses)
    eval_acc(model, val_dataloader)

if __name__ == "__main__":
   main()
   
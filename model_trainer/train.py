import Lightning as L
from model_trainer.models import model
from model_trainer.datasets.data_loader import load_data

def main():
    train_data = load_data()
    trainer = L.Trainer()
    trainer.fit(model=model, train_data=train_data)
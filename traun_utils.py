import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_generater import AudioGenerator
from torch.nn import CTCLoss
from models import SimpleRNNModel
import os

class CTCModel(nn.Module):
    def __init__(self, input_to_softmax):
        super(CTCModel, self).__init__()
        self.input_to_softmax = input_to_softmax
        self.ctc_loss = CTCLoss(blank=0, reduction='sum')

    def forward(self, x, labels, input_lengths, label_lengths):
        output_lengths = torch.IntTensor([x.size(1)] * x.size(0))
        logits = self.input_to_softmax(x)
        loss = self.ctc_loss(logits, labels, output_lengths, label_lengths)
        return loss

def train_model(input_to_softmax, 
                pickle_path,
                save_model_path,
                train_json='train_corpus.json',
                valid_json='valid_corpus.json',
                minibatch_size=20,
                spectrogram=True,
                mfcc_dim=13,
                optimizer=optim.SGD,
                learning_rate=0.02,
                decay=1e-6,
                momentum=0.9,
                nesterov=True,
                clipnorm=5,
                epochs=20,
                verbose=1,
                sort_by_duration=False,
                max_duration=10.0):
    
    # Create a class instance for obtaining batches of data
    audio_dataset_train = AudioGenerator(json_file=train_json, spectrogram=spectrogram, mfcc_dim=mfcc_dim, max_duration=max_duration, sort_by_duration=sort_by_duration)
    audio_dataset_valid = AudioGenerator(json_file=valid_json, spectrogram=spectrogram, mfcc_dim=mfcc_dim, max_duration=max_duration, sort_by_duration=sort_by_duration)

    # Create data loaders
    train_loader = DataLoader(audio_dataset_train, batch_size=minibatch_size, shuffle=True)
    valid_loader = DataLoader(audio_dataset_valid, batch_size=minibatch_size, shuffle=False)

    # Instantiate the CTCModel
    model = CTCModel(input_to_softmax)

    # Set up optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=decay, momentum=momentum, nesterov=nesterov)
    
    # Make results/ directory, if necessary
    if not os.path.exists('results'):
        os.makedirs('results')

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets, input_lengths, label_lengths in train_loader:
            optimizer.zero_grad()
            loss = model(inputs, targets, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for inputs, targets, input_lengths, label_lengths in valid_loader:
                loss = model(inputs, targets, input_lengths, label_lengths)
                valid_loss += loss.item()

        # Print the training and validation loss for each epoch
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader)}, Valid Loss: {valid_loss / len(valid_loader)}")

    # Save the PyTorch model
    torch.save(model.state_dict(), 'results/' + save_model_path)

    print("Training completed.")

if __name__ == "__main__":
    # Assuming you have your input_to_softmax model defined in PyTorch
    input_to_softmax = CTCModel(nn.Module)  # Replace with your PyTorch model
    train_model(input_to_softmax, pickle_path='your_pickle_path.pkl', save_model_path='your_model.pth')

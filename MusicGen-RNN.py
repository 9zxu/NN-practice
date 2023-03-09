import torch
import torch.nn as nn
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message
from music21 import *
import os
import sys
print(sys.executable)


# Define hyperparameters
input_size = 1
hidden_size = 128
num_layers = 2
sequence_length = 100
num_epochs = 1000
batch_size = 64
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MIDI files
midi_files = []

for file in os.listdir('midi_files'):
    if file.endswith('.mid'):
        midi_files.append(file)

# Create dictionary of notes to integers
notes = []
for file in midi_files:
    midi = MidiFile(os.path.join('midi_files', file))
    for msg in midi.play():
        if msg.type == 'note_on':
            note = msg.note
            if note not in notes:
                notes.append(note)

note_to_int = dict(zip(notes, range(len(notes))))

# Define the RNN model


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, len(notes))

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden


# Initialize the model and optimizer
model = RNN(input_size, hidden_size, num_layers)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(num_epochs):
    for file in midi_files:
        midi = MidiFile(os.path.join('midi_files', file))
        notes = []
        for msg in midi.play():
            if msg.type == 'note_on':
                note = msg.note
                notes.append(note)
        input_seq = [note_to_int[note] for note in notes]
        input_seq = np.array(input_seq)
        num_batches = len(input_seq) // sequence_length
        input_seq = input_seq[:num_batches*sequence_length]
        input_seq = input_seq.reshape(num_batches, sequence_length)
        input_seq = torch.tensor(input_seq, dtype=torch.float32)
        input_seq = input_seq.unsqueeze(2).to(device)
        target_seq = torch.roll(input_seq, -1, dims=1)
        target_seq[:, -1, :] = input_seq[:, 0, :]
        target_seq = target_seq.view(-1).long()
        hidden = None
        for i in range(num_batches):
            x = input_seq[i*batch_size:(i+1)*batch_size]
            y = target_seq[i*batch_size:(i+1)*batch_size]
            output, hidden = model(x, hidden)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
    if epoch % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
              1, num_epochs, loss.item()))

# Generate new MIDI file


def generate_music(model, note_to_int):
    model.eval()
    with torch.no_grad():
        notes = []
        note = np.random.choice(list(note_to_int.keys()))

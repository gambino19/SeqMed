# -*- coding: utf-8 -*-
"""
Main script for SeqMed training Many of the methods for training directly 
come from referenced SeqGAN repo (See https://github.com/suragnair/seqGAN)
"""

import random

import dill
import torch
import torch.nn as nn

from modules import Generator, Discriminator, unencode

def weights_init(m):
    """ 
    Normal Weights initialization per each module in a model
    for more details, seehttps://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_generator_MLE(gen, gen_opt, epochs):
    """
    Max Likelihood Estimate pre-training for the generator
    
    Notes:
        Structure is same as what is found in reference, although
        disregards oracle component (moreso for SeqGan experiment than architecture
        in whole) and makes necessary adjustments
        
    Keyword Arguments:
        gen (nn.Module): Generator defined in modules.py
        gen_opt (torch.optim): Generator optimizer
        epochs (int): Epochs for training; Default: EPOCHS
    
    Reference:
        https://github.com/suragnair/seqGAN
    """
    for epoch in range(epochs):
        total_loss = 0

        for patient in train_set:
            gen_opt.zero_grad()
            loss = gen.batch_loss(patient)
            loss.backward()
            gen_opt.step()
            
            total_loss += loss.data.item()
            
        print(f"Loss at Generator epoch {epoch+1}: {total_loss}")
        
def train_generator_PG(gen, gen_opt, disc, epochs, samples=100):
    """
    Policy gradient training for generator during adversarial training
    
    Notes:
        Structure is same as what is found in reference, although
        disregards oracle component (moreso for SeqGan experiment than architecture
        in whole) and makes necessary adjustments
        
    Keyword Arguments:
        gen (nn.Module): Generator object defined in modules.py
        gen_opt (torch.optim): Generator optimizer
        disc (nn.Module): Disciminator object defined in modules.py
        epochs (int): Epochs for training; Default: EPOCHS
        samples (int): Sample number for generator sampling; Default: SAMPLES
    
    Reference:
        https://github.com/suragnair/seqGAN
    """
    
    sample_idx = random.sample(range(len(train_set)), samples)
    positive_data = [train_set[i] for i in sample_idx] # All Codes
    data = [unencode(gen.sample(positive_data[i])[1]) for i in range(samples)]
    
    for epoch in range(epochs):
        loss = 0.0
        for i in range(samples):
            rewards = disc.batch_classify([data[i]])
            loss += gen.batch_policy_gradient(positive_data[i], rewards.item())
            loss.backward()
            gen_opt.step()
        print(f"Loss at Epoch {epoch}: {loss}")
        

def train_discriminator(disc, disc_opt, gen, epochs, samples=100):
    """
    Discriminator Training (pre-training + during adversarial process)
    
    Notes (verbatim from https://github.com/suragnair/seqGAN):
        Training the discriminator on real_data_samples (positive) and generated 
        samples from generator (negative). Samples are drawn `samples` times, and the 
        discriminator is trained for `epochs` epochs.
    
    Keyword Arguments:
        disc (nn.Module): Disciminator object defined in modules.py
        disc_opt (torch.optim): Discriminator optimizer
        gen (nn.Module): Generator object defined in modules.py
        epochs (int): Epochs for training; Default: EPOCHS
        samples (int): Sample number for generator sampling; Default: SAMPLES
    """

    # generating a small validation set before training (using oracle and generator)
    sample_idx = random.sample(range(len(train_set)), samples)
    positive_data = [train_set[i] for i in sample_idx] # All Codes
    
    negative_data = [unencode(gen.sample(positive_data[i])[1]) for i in range(samples)] # Already 1 0 vector
    positive_data = [code[2] for patient in positive_data for code in patient] # Only medicine vector, not encoded
    data = negative_data + positive_data
    labels = [0 for i in range(samples)] + [1 for i in range(samples)]    
    
    sample_idx = random.sample(range(2*samples), 2*samples)
    data = [data[i] for i in sample_idx]
    labels = [labels[i] for i in sample_idx]
    
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        for patient in range(2*samples):
            loss = 0.0
            disc_opt.zero_grad()
            output = disc.batch_classify([data[patient]])
            loss += disc.batch_loss([data[patient]], torch.tensor(labels[patient]))
            loss.backward()
            disc_opt.step()
            
            total_loss += loss.data.item()
            total_accuracy += torch.sum((output>0.5)==(labels[patient]>0.5)).data.item()
            
        print(f"Loss at Discriminator epoch {epoch+1}: {total_loss}")
        print(f"Accuracy at Discriminator epoch {epoch+1}: {total_accuracy/(2*samples)}")

if __name__ == '__main__':
    
    # Model Train conditions
    EPOCHS = 30 # In seq-med, convergence was claimed to be within 20 iterations
    SAMPLES = 200
    
    # Loading pre-processed training data as done by GAMENet
    # See https://github.com/sjy1203/GAMENet/tree/master/data for pkl file
    data = dill.load(open('./data/records_final.pkl', 'rb'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Finding Vocabulary size for diagnosis, procedure, and medicine codes
    vocab_diagnosis = len(set([data[i][j][0][k] for i in range(len(data)) for j in range(len(data[i])) for k in range(len(data[i][j][0]))])) + 2
    vocab_procedure = len(set([data[i][j][1][k] for i in range(len(data)) for j in range(len(data[i])) for k in range(len(data[i][j][1]))])) + 2
    vocab_medicine = len(set([data[i][j][2][k] for i in range(len(data)) for j in range(len(data[i])) for k in range(len(data[i][j][2]))])) + 2
    # Addition of two to all vocab size since values start at 2-..., so this gets shape for empty torch tensors correct without
    # needing to shift all values over

    # Data manipulation for training
    random.seed(181)
    tmp = data.copy()
    random.shuffle(tmp)
    
    train_set = tmp[:int(len(data)*0.8)]
    test_set = tmp[int(len(data)*0.8):]    

    gen = Generator(vocab_diagnosis, vocab_procedure, vocab_medicine)
    gen.apply(weights_init)
    disc = Discriminator(vocab_medicine)
    disc.apply(weights_init)

    if device.type == 'cude':
        gen = gen.cuda()
        disc = disc.cuda()

    # Generator Training before adversarial process
    print('Starting Generator Training...')
    print('-------------------------------')
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=1e-3)
    train_generator_MLE(gen, gen_optimizer, epochs=EPOCHS)

    # Discriminator Training before adversarial process
    print('Starting Discriminator Training...')
    print('-------------------------------')
    disc_optimizer = torch.optim.Adagrad(disc.parameters())
    train_discriminator(disc, disc_optimizer, gen, epochs=EPOCHS, samples=SAMPLES)

    # Adversarial Training
    print('Starting Adversarial Training...')
    print('---------------------------------')
    for epoch in range(10):
        print('Adversarial Training Generator : ')
        print('--------------------------------------')
        train_generator_PG(gen, gen_optimizer, disc, epochs=EPOCHS, samples=SAMPLES)

        # TRAIN DISCRIMINATOR
        print('Adversarial Training Discriminator : ')
        print('--------------------------------------')
        train_discriminator(disc, disc_optimizer, gen, epochs=EPOCHS, samples=SAMPLES)
"""
SeqMed Modules

All modules implemented follow the guidelines and parameters found in:
S. Wang,  "SeqMed: Recommending Medication Combination with Sequence Generative Adversarial Nets," in 2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Seoul, Korea (South), 2020 pp. 2664-2671.
doi: 10.1109/BIBM49941.2020.9313196
keywords: {medical diagnostic imaging;task analysis;data mining;reinforcement learning;medical services;gallium nitride;predictive models}
url: https://doi.ieeecomputersociety.org/10.1109/BIBM49941.2020.9313196

Select modules (namely, the Generator and the discriminator), reference prior
SeqGAN work (see https://github.com/suragnair/seqGAN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClinincalInformationExtraction(nn.Module):
    """
    Clinical Information Extraction Module 
    Embed Medical records and extract clinical information
    
    Notes:
        General forwawrd-pass flow for Module (as described by SeqMed)
        1. Get embeddings through projecting input codes through Embedding matricies 
        2. Get feature vectors through passing embeddings through CNN
        3. Get representation vectors after passing feature vectors through linear and sigmoid
        
        Some liberties were taken in this implementation of the model vs what was described
        in the paper - primarily, their use of the word "hidden size" when describing the 
        architecture does not really fall in scope of standard nomenclature. Therefore,
        it was interpretted to mean the size of the input/output of the hidden layers
    
    Keyword Arguments:
        num_embeddings (int): Number of embeddings for nn.Embedding; also known as vocab size
        embedding_dim (int): Embedding dimension for nn.Embedding; Default: 100
        cnn (nn.Sequential): 3-layer Convolutional Neural Network with "hidden size" set to 128
        linear (nn.Linear): Linear layer
        sigmoid (nn.Sigmoid): Sigmoid activation function
        
    Attributes:
        num_embeddings (int): Number of embeddings for nn.Embedding; also known as vocab size
        embedding_dim (int): Embedding dimension for nn.Embedding
    """
    
    def __init__(self, num_embeddings, embedding_dim=100): # In GAMENet, num_embeddings == vocab_size
        
        super(ClinincalInformationExtraction, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, 
                                      embedding_dim=embedding_dim)
        
        self.cnn = nn.Sequential( # No BatchNorm used since process only describes one sample per batch
            # First Layer
            nn.Conv1d(in_channels=embedding_dim, 
                      out_channels=128, 
                      kernel_size=1), # Bias Default Trye
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=1),
            # Second Layer
            nn.Conv1d(in_channels=128, 
                      out_channels=128, 
                      kernel_size=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=1),
            # Third Layer
            nn.Conv1d(in_channels=128, 
                      out_channels=128, 
                      kernel_size=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=1),
            )
        
        self.linear = nn.Linear(128, 100) # Bias Default True
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, codes):
        """
        Model Forward Pass definiton
        
        Notes:
            As defined in the paper, the forward pass of the model only uses one sample
        
        Keyword Arguments:
            code (list): Multi-hot vector at the t-th visit that shows the medical codes at the recorded t-th visit
            
        Returns:
            torch.tensor: Extracted representation of code data
            
        Examples:
            >>> torch.manual_seed(2)
            >>> cie = ClinincalInformationExtraction(num_embeddings=vocab_diagnosis, embedding_dim=100)
            >>> cie([[0,1], [1,2]])
            tensor(0.3606, grad_fn=<SelectBackward0>)
        """
        
        embeddings = []
        for code in codes:
            emb = self.embedding(torch.tensor(code))
            emb_mean = emb.mean(dim=0).unsqueeze(dim=0) # One-sample every batch, as done in GAMENet as welll
            embeddings.append(emb_mean)
        
        embeddings = torch.cat(embeddings, dim=0).unsqueeze(dim=0)
        
        # Since 1D has Cin as second dim, need to change axis
        feature_vectors = torch.permute(embeddings, (0, 2, 1))
        feature_vectors = self.cnn(feature_vectors)
        feature_vectors = torch.permute(feature_vectors, (0, 2, 1))
        
        representation_vector = self.linear(feature_vectors)
        representation_vector = self.sigmoid(representation_vector)
        
        return representation_vector.squeeze(0)
    
class HealthStatusUpdate(nn.Module):
    """
    Health Status Update Module
    Updates the health status dynamically with a vector of predicted medicines
    
    Notes:
        Every time a new medicine is predicted bt the generator, the vector
        for predicted medicines (self.predicted_medicines) will be updated and
        will run through another forward pass
    
        Some liberties were taken in this implementation of the model vs what was described
        in the paper - the use of a "project matrix" was unclearly defined. Namely, 
        to what dimensions do we project upon and with what other vector, are we
        projecting or unprojecting since we are going onto a higher dimensional space, etc.
        This implementation treats that projection (self.project_matrix) as an embedding,
        which means as the model learns, so to will the projection ability
        
    Keyword Arguments:
        vocab_diagnosis (int): Vocabulary size for MIMIC-III diagnosis data
        vocab_procedure (int): Vocabulary size for MIMIC-III procedure data
        vocab_medicine (int): Vocabulary size for MIMIC-III medicine data
        embedding_dim (int): Embedding dimension for nn.Embedding; Default: 100
    
    Attributes:
        vocab_diagnosis (int): Vocabulary size for MIMIC-III diagnosis data
        vocab_procedure (int): Vocabulary size for MIMIC-III procedure data
        vocab_medicine (int): Vocabulary size for MIMIC-III medicine data
        embedding_dim (int): Embedding dimension for nn.Embedding
        projection_matrix (nn.Embedding): Projection matrix as an embeding; Shape (vocab_medicine, 2*embedding_dim)
        cie_diagnosis (nn.Module) ClinincalInformationExtraction for diagnosis data
        cie_procedure (nn.Module) ClinincalInformationExtraction for procedure data
    """
    
    def __init__(self, vocab_diagnosis, vocab_procedure, vocab_medicine, embedding_dim=100):
        
        super(HealthStatusUpdate, self).__init__()
        
        self.vocab_diagnosis = vocab_diagnosis
        self.vocab_procedure = vocab_procedure
        self.vocab_medicine = vocab_medicine
        self.embedding_dim = embedding_dim
        
        # Health Status Update variables as defined in paper are added in as comments
        self.projection_matrix = nn.Embedding(vocab_medicine, 2*embedding_dim) # W_g
        
        # Components of HealthStatusUpdate
        self.cie_diagnosis = ClinincalInformationExtraction(num_embeddings=vocab_diagnosis)
        self.cie_procedure = ClinincalInformationExtraction(num_embeddings=vocab_procedure)
    
    def forward(self, codes_diagnosis, codes_procedure, predicted_medicines):
        """
        Model Forward Pass definiton
        
        Notes:
            Forward pass for HealthStatusUpdate will be run whenever a new 
            medicine (as predicted by our generator) is updated in the 
            self.predicted_medicine.
        
        Keyword Arguments:
            code_diagnosis (list): Multi-hot vector at the t-th visit that shows the diagnosis codes at the recorded t-th visit
            code_procedure (list): Multi-hot vector at the t-th visit that shows the procedure codes at the recorded t-th visit
            predicted_medicine (torch.tensor): 1D one-hot encoding of predicted medicine codes 
            
        Returns:
            torch.tensor: Attention mechanism weights
            torch.tensor: Attention mechanism alignment
        
        Examples:
            >>> torch.manual_seed(2)
            >>> hsu = HealthStatusUpdate(vocab_diagnosis=vocab_diagnosis, 
                                         vocab_procedure=vocab_procedure,
                                         vocab_medicine=vocab_medicine)
            >>> hsu([[1,2], [1]], [[1], [12, 3]], [[0,1,5], [1, 5, 6]])
            tensor(torch.tensor(...), torch.tensor(...))
        """
        
        # Concatenated Health Status update module
        z_d = self.cie_diagnosis(codes_diagnosis)
        z_p = self.cie_procedure(codes_procedure)
        Z = torch.cat((z_d, z_p), dim=1)
        
        gW_g = self.projection_matrix(predicted_medicines).detach() # Alternative name for predicted medicine (as done in generator): g
        
        a = F.log_softmax(Z @ gW_g.T, dim=1) #Changed to log_softmax
        h = a.T @ Z + gW_g
        
        return a, h
    
class Generator(nn.Module):
    """
    GRU Generator for SeqMed
    
    Notes:
        Implemented GAN generator mostly follows implementation described
        by author, but also takes some components/methodologies of other
        publically available GANs for Sequential Modeling. See References for
        more details
    
    Keyword Arguments:
        vocab_diagnosis (int): Vocabulary size for MIMIC-III diagnosis data
        vocab_procedure (int): Vocabulary size for MIMIC-III procedure data
        vocab_medicine (int): Vocabulary size for MIMIC-III medicine data
        hidden_size (int): GRU hidden size; Default: 32
        embedding_dim (int): Embedding dimension for nn.Embedding; Default: 100
    
    Attributes:
        vocab_diagnosis (int): Vocabulary size for MIMIC-III diagnosis data
        vocab_procedure (int): Vocabulary size for MIMIC-III procedure data
        vocab_medicine (int): Vocabulary size for MIMIC-III medicine data
        hidden_size (int): GRU hidden size
        embedding_dim (int): Embedding dimension for nn.Embedding
        hsu (nn.Module): Health Status Update Module for Generator
        gru (nn.GRU): GRU unit for Generator; input_size=vocab_medicine, hidden_size=hidden_size
        linear (nn.Linear): Linear Unit after GRU; in_features=hidden_size, out_features=vocab_medicine
        sigmoid (nn.Sigmoid): Sigmoid Activation after linear
        
    References:
        https://github.com/suragnair/seqGAN
    """
    
    
    def __init__(self, vocab_diagnosis, vocab_procedure, vocab_medicine, 
                 hidden_size=32, embedding_dim=100):
        
        super(Generator, self).__init__()
        
        self.vocab_diagnosis = vocab_diagnosis
        self.vocab_procedure = vocab_procedure
        self.vocab_medicine = vocab_medicine
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        
        self.hsu = HealthStatusUpdate(vocab_diagnosis, vocab_procedure, vocab_medicine)
        
        self.gru = nn.GRU(vocab_medicine, hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_medicine)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, xt, h, g):
        """
        Model Forward Pass definiton for single sequence
        
        Notes:
            Forward pass of generator is done through the batch_loss method,
            which will calculate loss across all sequences x for xt in T
            
            After pass through GRU, will add medicine to predicted medicines
            vector. G will only update if the selected medicine is not
            already seen in the vector
        
        Keyword Arguments:
            xt (torch.tensor): Health Status representation at sequence t
            h (torch.tensor): Hidden States at sequence t-1
            g (torch.tensor): Predicted mediciunes up to t-1
            
        Returns:
            torch.tensor: Output token distrubution form p(y|x1....t)
            torch.tensor: Hidden states at t
            torch.tensor: Predicted medicines at t
        """
        
        output, h_t = self.gru(xt.unsqueeze(0), h)
        zh_t = self.sigmoid(self.linear(h_t)).view(-1)
        max_prob = zh_t.max(dim=0)
        if g[max_prob.indices].item() != 1:
            g.put_(max_prob.indices, torch.tensor(1))
        
        # Prior method of TopK selections would infinitely 
        # grow medicine recommendation vector to include even medicines not needed
        # options = torch.topk(zh_t, self.vocab_medicine, dim=0)
        # for i in options.indices:
        #     if g[i].item() != 1.0:
        #         g[i] = 1.0
        #         break
        
        return zh_t, h_t, g
    
    def batch_policy_gradient(self, sequences, reward):
        """
        Apply Policy Gradient (sometimes listed as PG) across sampled batch
        in generator
        
        Notes:
            For the reward, the REINFORCE algorithm is selected, where then
            the estimated probability of the disrimintor is used as the reward
        
        Keyword Arguments:
            sequences (list): Codes for diagnosis, procedure, and medicines in data
            reward (torch.tensor): Reward  (see notes)
                
        Returns:
            torch.tensor: Loss across generator batch for policy gradient
        """
        
        loss = torch.tensor(0.0, requires_grad=True)
        sequences_length = len(sequences)
        
        codes_diagnosis = [code[0] for code in sequences]
        codes_procedure = [code[1] for code in sequences]
        # codes_medicine = [code[2] for code in sequences]
        
        h0 = torch.zeros(1,1,self.hidden_size)
        g =  torch.zeros((self.vocab_medicine,), dtype=int) # predicted_medicines
        
        for i in range(sequences_length):
            xt, _ = self.hsu([codes_diagnosis[i]], [codes_procedure[i]], g)
            zh_t, h_n, g = self.forward(xt, h0, g)
            loss = loss + (g*reward).sum()
            
        return loss
        

    def batch_loss(self, sequences):
        """
        Calculate loss across all sequences in a sampled batch
        
        Notes:
            Method directly takes influence to what was done in the reference
            
            In reference to the paper, this portion covers loss L_n through
            use of nn.NLLLoss. Total loss L will be calculated across samples on training
        
        Keyword Arguments:
            sequences (list): Codes for diagnosis, procedure, and medicines in data
        
        Examples:
            >>> torch.manual_seed(2)
            >>> data = dill.load(open(f'./data/records_final.pkl', 'rb')) # GAMENet Data
            >>> sequences = data[0]
            >>> gen = Generator(vocab_diagnosis, vocab_procedure, vocab_medicine)
            >>> gen.batch_loss(data[0])
            tensor(-0.9745, grad_fn=<AddBackward0>)
        """
        
        loss_fn = nn.NLLLoss()
        loss = 0.0
        sequences_length = len(sequences)
        
        codes_diagnosis = [code[0] for code in sequences]
        codes_procedure = [code[1] for code in sequences]
        codes_medicine = [code[2] for code in sequences]
        
        h0 = torch.zeros(1,1,self.hidden_size)
        g =  torch.zeros((self.vocab_medicine,), dtype=int) # predicted_medicines
        
        for i in range(sequences_length):
            xt, _ = self.hsu([codes_diagnosis[i]], [codes_procedure[i]], g)
            zh_t, h_n, g = self.forward(xt, h0, g)
            h0 = h_n
            loss += loss_fn(zh_t, encode([codes_medicine[i]], self.vocab_medicine).view(-1))
        
        return loss
    
    def sample(self, sequences):
        """
        Generate sample medicine vector from a given sequence
        
        Keyword Arguments:
            sequencess (list): Codes for diagnosis, procedure, and medicines in data
            
        Returns:
            torch.tensor: Probability distributions for output token distribution
            torch.tensor: Predicted medicine vector
        """
        
        h0 = torch.zeros(1, 1, self.hidden_size)
        g =  torch.zeros((self.vocab_medicine,), dtype=int) # predicted_medicines
        
        codes_diagnosis = [code[0] for code in sequences]
        codes_procedure = [code[1] for code in sequences]
        # codes_medicine = [code[2] for code in sequences] # Medicine vector not needed for sampling
        
        for i in range(len(sequences)):
            for _ in range(self.vocab_medicine):
                xt, _ = self.hsu([codes_diagnosis[i]], [codes_procedure[i]], g)
                zh_t, h_n, g = self.forward(xt, h0, g)
                h0 = h_n
            
        return zh_t, g
        
class Discriminator(nn.Module):
    """
    GRU Discriminator for SeqMed
    
    Notes:
        Implemented GAN discriminator mostly follows implementation described
        by author, but also takes some components/methodologies of other
        publically available GANs for Sequential Modeling. See References for
        more details
    
    Keyword Arguments:
        vocab_medicine (int): Vocabulary size for MIMIC-III medicine data
        hidden_size (int): GRU hidden size; Default: 64
        dropout (int): See p in nn.Dropout; Default: 0.2 
    
    Attributes:
        vocab_medicine (int): Vocabulary size for MIMIC-III medicine data
        hidden_size (int): GRU hidden size
        gru (nn.GRU): GRU unit for Discriminator; input_size=vocab_medicine, hidden_size=hidden_size
        tanh (nn.Tanh): Tanh activation function for discriminator
        linear (nn.Linear): Linear Unit for classification; in_features=hidden_size, out_features=1
        sigmoid (nn.Sigmoid): Sigmoid Activation after linear
        
    References:
        https://github.com/suragnair/seqGAN
    """

    def __init__(self, vocab_medicine, hidden_size=64, dropout=0.2):
        
        super(Discriminator, self).__init__()
        
        self.vocab_medicine = vocab_medicine
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(vocab_medicine, hidden_size=hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, code_medicine, hidden):
        """
        Model Forward Pass definiton for single sequence
        
        Notes:
            Forward pass of generator is done through the batch_loss or
            batch_classify, as this method primarily handles model modules
            and functionals
        
        Keyword Arguments:
            code_medicine (torch.tensor): One-hot encoded medicines to classify
            hidden (torch.tensor): Hidden States of sequence
        
        Returns:
            torch.tensor: Classification probability for code_medicine
        """
        
        _, hidden = self.gru(code_medicine.float(), hidden.float())
        output = self.tanh(hidden)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        
        return output
        
    def batch_classify(self, code_medicine):
        """
        Classify along batch
        
        Notes:
            Since the process of SeqMed (and alternatively, other works such\
            such as GAMENet) only rely on one sample across the batch,
            this will return a single classification probability. Could be potentially
            easily extended to multuiple codes
        
        Keyword Arguments:
            code_medicine (torch.tensor): One-hot encoded medicines to classify
        
        Returns:
            torch.tensor: Classification probability for code_medicine
        """
        
        hidden = torch.zeros((1, len(code_medicine), self.hidden_size))
        code_medicine = encode(code_medicine, self.vocab_medicine)
        output = self.forward(code_medicine, hidden)
        
        return output.view(-1)
        
    def batch_loss(self, code_medicine, target):
        """
        Get loss across batch
        
        Notes:
            Since the process of SeqMed (and alternatively, other works such\
            such as GAMENet) only rely on one sample across the batch,
            this will return a single loss. Could be potentially
            easily extended to multuiple codes
            Loss function used for this is BCELoss
        
        Keyword Arguments:
            code_medicine (torch.tensor): One-hot encoded medicines to classify
            target (torch.tensor): Target one-hot encoded medicine vector
        
        Returns:
            torch.tensor: Loss for code_medicine against target
        """
        
        loss_fn = nn.BCELoss(reduction='sum')
        
        hidden = torch.zeros((1, len(code_medicine), self.hidden_size))
        code_medicine = encode(code_medicine, self.vocab_medicine)
        output = self.forward(code_medicine, hidden)
        
        target = torch.tensor(target).view(1,-1,1).float()
        
        return loss_fn(output, target)
        
def encode(codes, vocab_size):
    """
    Encode code by index to some one-hot vector
    
    Notes:
        Primarily used for medicine vector prediction
        
    Keyword Arguments:
        codes (list): Codes to encode to one-hot vector
        vocab_size (int): Size of vocab; determines the size of the one-hot vector
    """
    one_hot = torch.zeros((1, len(codes), vocab_size), dtype=int)
    for i, code in enumerate(codes):
        one_hot[0,i,:].put_(torch.tensor(code), torch.tensor([1 for _z in range(len(code))]))
    return one_hot

def unencode(codes):
    """
    Unencode code by one-hot vector to some list of indices
    
    Notes:
        Primarily used for medicine vector prediction
        
    Keyword Arguments:
        codes (list): Codes to unencode to list of indices
    """
    mask = codes == 1
    return mask.nonzero().view(-1).tolist()
    
    

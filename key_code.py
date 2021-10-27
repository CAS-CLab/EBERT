import torch
import torch.nn as nn
from torch.autograd import Variable

class GumbelSoftmax(nn.Module):
    '''
        gumbel softmax gate.
    '''
    def __init__(self, eps=1):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
    
    def gumbel_sample(self, template_tensor, eps=1e-8):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor+eps)-torch.log(
                                          1-uniform_samples_tensor+eps)
        return gumble_samples_tensor
    
    def gumbel_softmax(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gsamples = self.gumbel_sample(logits.data)
        logits = logits + Variable(gsamples)
        soft_samples = self.sigmoid(logits / self.eps)
        return soft_samples, logits
    
    def forward(self, logits):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        if not self.training:
            out_hard = (logits>=0).float()
            return out_hard
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard

class BertSelfAttention(nn.Module):
    def __init__(self, config, layer_id=0):

        self.predictor = nn.Sequential(
                            nn.Linear(config.hidden_size, 64),
                            nn.BatchNorm1d(64, eps=config.layer_norm_eps),
                            nn.ReLU(),
                            nn.Linear(64, config.num_attention_heads),
                        )
        self.gumbel = GumbelSoftmax()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        ...
        # first layer use mean of hidden_states as input of predictor
        if self.layer_id == 0:
            logits = self.predictor(torch.mean(hidden_states, dim=1))
        else:
            logits = self.predictor(hidden_states[:, 0, :])
        dynamic_head_mask = self.gumbel(logits)
        
        attention_probs = attention_probs * dynamic_head_mask

        ...

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.predictor = nn.Sequential(
                                nn.Linear(config.hidden_size, 64), 
                                nn.BatchNorm1d(64, eps=config.layer_norm_eps),
                                nn.ReLU(), 
                                nn.Linear(64, config.intermediate_size),
                            )      
        self.gumbel = GumbelSoftmax()

    def forward(self, hidden_states):
        ...

        logits = self.predictor(hidden_states[:, 0, :])
        dynamic_ffn_mask = self.gumbel(logits)
        hidden_states = hidden_states * dynamic_ffn_mask

        ...
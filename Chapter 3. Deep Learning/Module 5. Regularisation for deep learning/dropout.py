import torch

class NNWithDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.8),
            # torch.nn.Linear(784, 516),
            # torch.nn.ReLU(),
            Dropout(p=0.5),
        )

    def forward(self, x):
        return self.layers(x)


def train(model):
    model.train()

def test():
    model.eval()

class Dropout(torch.nn.Module):
    def __init__(self, p):
        self.p = p
        super().__init__()

    def forward(self, x):
        prob_vect = torch.ones_like(x) * self.p
        mask = torch.bernoulli(prob_vect)
        print(mask)
        x = x * mask
        return x

batch_size = 4
x = torch.rand(batch_size, 3)

nnd = NNWithDropout()
# nnd.train()
print(x)
result = nnd(x)
print(result)

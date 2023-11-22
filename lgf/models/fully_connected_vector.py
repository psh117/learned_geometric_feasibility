import torch.nn as nn

class FullyConnectedVector(nn.Module):
    def __init__(self, 
                 layer_dims=[12, 512, 512, 5],
                 dropouts=[None, None, None],
                #  droupouts=[nn.Dropout(p=0.5), nn.Dropout(p=0.5), None],
                 activations=[nn.ReLU(), nn.ReLU(), nn.Sigmoid()]):
        super(FullyConnectedVector, self).__init__()
        
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

        layers = []
        prev_dim = layer_dims[0] # input dim

        # assert(len(layer_dims) == len(dropouts) + 1, "layer_dims and dropouts must have the same length" + str(len(layer_dims)) + " " + str(len(dropouts)))
        # assert(len(layer_dims) == len(activations) + 1, "layer_dims and activations must have the same length" + str(len(layer_dims)) + " " + str(len(activations)))

        for layer_dim, dropout, activation in zip(layer_dims[1:], dropouts, activations):
            layers.append(nn.Linear(prev_dim, layer_dim))

            if dropout is not None:
                layers.append(dropout)

            if activation is not None:
                layers.append(activation)
                
            prev_dim = layer_dim

        self.net = nn.Sequential(*layers)

        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)

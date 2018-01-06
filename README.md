
### Matlab implementation of some recurrent neural networks (RNNs) as follows
* Vanilla RNN 
* Gated Recurrent Unit ([GRU](https://arxiv.org/abs/1406.1078)) 
* Long Short-Term Memory ([LSTM](https://arxiv.org/abs/1303.5778)) 
* Multiplicative Integration RNN ([MIRNN](https://arxiv.org/abs/1606.06630)) 
* Control Gate based RNN ([CGRNN](https://dl.acm.org/citation.cfm?doid=2964284.2967249))
* Self-Gated RNN ([SGRNN](https://dl.acm.org/citation.cfm?doid=3126686.3126764))

These codes were written a long time ago when I started with deep learning, but they include some codes for computing gradients which are often absent in current Python codes of DL models. So, I think it is worthy to put them here for reference. :+1: 

> The codes are only for classification task in which RNN type is one direction with one or two layers, and the decision is based on the last hidden state. Input is in *cell array* format , each component in a cell corresponds to a timestep.

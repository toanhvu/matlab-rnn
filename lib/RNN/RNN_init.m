function theta = RNN_init(input_size, hidden_size, output_size)
avg = 0;
sigma = 1;

W1 = 0.1 * normrnd(avg,sigma, hidden_size, input_size);
% W2 = iden_scalar * eye(hidden_size);
W2 = 0.01 * normrnd(avg,sigma, hidden_size, hidden_size);
W3 = 0.1 * normrnd(avg,sigma, output_size, hidden_size);
Bi = zeros(hidden_size, 1);
Bo = zeros(output_size, 1);
theta = [W1(:); W2(:); W3(:); Bi(:); Bo(:)];
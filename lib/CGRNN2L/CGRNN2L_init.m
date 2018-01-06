function theta = CGRNN2L_init(input_size, ha_size, hb_size, output_size)
avg = 0;
sigma = 1;

Wa1 = 0.1 * normrnd(avg,sigma, ha_size, input_size);
Wa2 = 0.1 * normrnd(avg,sigma, ha_size, ha_size); 
% Wa2 = eye(ha_size);
Va1 = 0.01 * normrnd(avg,sigma, ha_size, input_size);
Va2 = 0.01 * normrnd(avg,sigma, ha_size, ha_size);
Bka = zeros(ha_size, 1);
Bza = zeros(ha_size, 1);

theta1 = [Wa1(:); Wa2(:); Va1(:); Va2(:); Bka(:); Bza(:)]; 

Wb1 = 0.1 * normrnd(avg,sigma, hb_size, ha_size);
Wb2 = 0.1 * normrnd(avg,sigma, hb_size, hb_size);
% Wb2 = eye(hb_size);
Vb1 = 0.01 * normrnd(avg,sigma, hb_size, ha_size);
Vb2 = 0.01 * normrnd(avg,sigma, hb_size, hb_size);
Bkb = zeros(hb_size, 1);
Bzb = zeros(hb_size, 1);
theta2 = [Wb1(:); Wb2(:); Vb1(:); Vb2(:); Bkb(:); Bzb(:)]; 

Wo = 0.1 * normrnd(avg,sigma, output_size, hb_size);
Bo = zeros(output_size, 1);
theta0 = [Wo(:); Bo(:)];

theta = struct;
theta.theta0 = theta0;
theta.theta1 = theta1;
theta.theta2 = theta2;


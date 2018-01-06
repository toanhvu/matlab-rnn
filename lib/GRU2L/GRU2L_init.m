function theta = GRU2L_init(input_size, ha_size, hb_size, output_size)
avg = 0;
sigma = 0.01;

Wa1 = normrnd(avg,sigma, ha_size, input_size);
Wa2 = normrnd(avg,sigma, ha_size, ha_size);
Ua1 = normrnd(avg,sigma, ha_size, input_size);
Ua2 = normrnd(avg,sigma, ha_size, ha_size);
Va1 = normrnd(avg,sigma, ha_size, input_size);
Va2 = normrnd(avg,sigma, ha_size, ha_size);
Bka = zeros(ha_size, 1);
Bza = zeros(ha_size, 1);
Bra = zeros(ha_size, 1);

theta1 = [Wa1(:); Wa2(:); Ua1(:); Ua2(:); Va1(:); Va2(:); Bka(:); Bza(:); Bra(:)]; 

Wb1 = normrnd(avg,sigma, hb_size, ha_size);
Wb2 = normrnd(avg,sigma, hb_size, hb_size);
Ub1 = normrnd(avg,sigma, hb_size, ha_size);
Ub2 = normrnd(avg,sigma, hb_size, hb_size);
Vb1 = normrnd(avg,sigma, hb_size, ha_size);
Vb2 = normrnd(avg,sigma, hb_size, hb_size);
Bkb = zeros(hb_size, 1);
Bzb = zeros(hb_size, 1);
Brb = zeros(hb_size, 1);
theta2 = [Wb1(:); Wb2(:); Ub1(:); Ub2(:); Vb1(:); Vb2(:); Bkb(:); Bzb(:); Brb(:)]; 


Wo = normrnd(avg,sigma, output_size, hb_size);
Bo = zeros(output_size, 1);
theta0 = [Wo(:); Bo(:)];

theta = struct;
theta.theta0 = theta0;
theta.theta1 = theta1;
theta.theta2 = theta2;
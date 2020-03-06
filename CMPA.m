clear all
close all

I_s = 0.01e-12; %amps
I_b = 0.1e-12; %amps
V_b = 1.3; %Volts
G_p = 0.1; %Ohms^-1

%% Part 1

V = linspace(-1.95,0.7,200);
I = I_s*(exp(1.2/0.025*V)-1) + G_p*V - I_b*(exp(-1.2/0.025*(V+V_b))-1);
I_rand = I + I.*(rand(200,1)*0.4-0.2)';


%% Part 2
p4_plot = polyfit(V,I,4);
p8_plot = polyfit(V,I,9);
f4_plot =  polyval(p4_plot,V);
f8_plot = polyval(p8_plot,V);

p4_log = polyfit(V,abs(I_rand),4);
p8_log = polyfit(V,abs(I_rand),9);
f4_log =  polyval(p4_log,V);
f8_log = polyval(p8_log,V);

%% Part 3

fo_AC = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff_AC = fit(V',I',fo_AC);
If_AC = ff_AC(V);

fo_ABC = fittype('A.*(exp(1.2*x/25e-3)-1) + B*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff_ABC = fit(V',I',fo_ABC);
If_ABC = ff_ABC(V);

fo_all = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff_all = fit(V',I',fo_all);
If_all = ff_all(V);

fo_AC_log = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff_AC_log = fit(V',abs(I_rand)',fo_AC_log);
If_AC_log = ff_AC_log(V);

fo_ABC_log = fittype('A.*(exp(1.2*x/25e-3)-1) + B*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff_ABC_log = fit(V',abs(I_rand)',fo_ABC_log);
If_ABC_log = ff_ABC_log(V);

fo_all_log = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff_all_log = fit(V',abs(I_rand)',fo_all_log);
If_all_log = ff_all_log(V);

%% Part 3 Neural Net
inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs;

inputs_log = V.';
targets_log = abs(I).';
hiddenLayerSize_log = 10;
net_log = fitnet(hiddenLayerSize);
net_log.divideParam.trainRatio = 70/100;
net_log.divideParam.valRatio = 15/100;
net_log.divideParam.testRatio = 15/100;
[net_log,tr_log] = train(net_log,inputs_log,targets_log);
outputs_log = net_log(inputs_log);
errors_log = gsubtract(outputs_log,targets_log);
performance_log = perform(net_log,targets_log,outputs_log);
view(net_log)
Inn_log = outputs_log;


%% Figures
figure;
%Polyfit Plot
subplot(3,2,1)
plot(V,I,'b',V,f4_plot,'r',V,f8_plot,'y')
title('polyfit')
xlabel('V')
ylabel('I')

%Polyfit Log Plot
subplot(3,2,2)
semilogy(V,abs(I_rand),'b',V,abs(f4_log),'r',V,abs(f8_log),'y')
title('polyfit')
xlabel('V')
ylabel('abs(I)')

%nonlinear fit
subplot(3,2,3)
plot(V,I,'b',V,If_AC','r',V,If_ABC','y',V,If_all','m')
title('non-linear fit')
xlabel('V')
ylabel('I')

%nonlinear log fit
subplot(3,2,4)
semilogy(V,abs(I_rand),'b',V,abs(If_AC_log'),'r',V,abs(If_ABC_log'),'y',V,abs(If_all_log'),'m')
title('non-linear fit')
xlabel('V')
ylabel('I')

%Neural Net fit
subplot(3,2,5)
plot(V,I,'b',V,Inn','r')
title('neural net')
xlabel('V')
ylabel('I')

%Neural Net log fit
subplot(3,2,6)
semilogy(V,abs(I_rand),'b',V,Inn_log','r')
title('neural net')
xlabel('V')
ylabel('I')
%%% Written by Chainplain 2023-7-17

sampleFreq = 5;
sampleDif  = 8;

MMS_ConEN = zeros(sampleFreq,sampleDif);
MUS_ConEN = zeros(sampleFreq,sampleDif);
MMS_Error = zeros(sampleFreq,sampleDif);
MUS_Error = zeros(sampleFreq,sampleDif);
MMS_UCB = zeros(sampleFreq,sampleDif);
MUS_UCB = zeros(sampleFreq,sampleDif);
MMS_DGram = zeros(sampleFreq,sampleDif);
MUS_DGram = zeros(sampleFreq,sampleDif);

for N = 1:sampleFreq 
    load('ConEn_learning_results'+ string(N-1) +'.mat')
    MMS_ConEN(N,:) = MMS;
    MUS_ConEN(N,:) = MUS;

    load('Error_learning_results'+ string(N-1) +'.mat')
    MMS_Error(N,:) = MMS;
    MUS_Error(N,:) = MUS;

    load('UCB_learning_results'+ string(N-1) +'.mat')
    MMS_UCB(N,:) = MMS;
    MUS_UCB(N,:) = MUS;

    load('DGram_learning_results'+ string(N-1) +'.mat')
    MMS_DGram(N,:) = MMS;
    MUS_DGram(N,:) = MUS;
end

figure;
set(gcf,'Position',[200,100,600,400]);
lw = 1;


Deepred = '#b4534b';
Deepgreen = '#6d8346';
Deepblue = '#426ab3';
Vio   = '#8552a1';

mean_MMS_DGram = mean(MMS_DGram);
mean_MMS_ConEN = mean(MMS_ConEN);
mean_MMS_Error = mean(MMS_Error);
mean_MMS_UCB = mean(MMS_UCB);

mean_MUS_DGram = mean(MUS_DGram);
mean_MUS_ConEN = mean(MUS_ConEN);
mean_MUS_Error = mean(MUS_Error);
mean_MUS_UCB = mean(MUS_UCB);

% grid on;
% h = gca;
% set(h,'LineWidth',1,'GridLineStyle','--','GridAlpha',1,'GridColor',[0.92,0.9,0.88])
subplot(5,2,1);
ba =boxchart(MMS_DGram);
ba.BoxFaceColor = Deepred;

subplot(5,2,3);
bb =boxchart(MMS_ConEN);
bb.BoxFaceColor = Deepgreen;

subplot(5,2,5);
bc = boxchart(MMS_Error);
bc.BoxFaceColor = Deepblue;

subplot(5,2,7);
bd = boxchart(MMS_UCB);
bd.BoxFaceColor = Vio;

subplot(5,2,9);
hold on;
plot(kernel_target_nums, mean_MMS_DGram,'o-', 'Linewidth',lw, 'MarkerFaceColor','w','color',Deepred);
plot(kernel_target_nums, mean_MMS_ConEN,'o-', 'Linewidth',lw, 'MarkerFaceColor','w','color',Deepgreen);
plot(kernel_target_nums, mean_MMS_Error,'o-', 'Linewidth',lw, 'MarkerFaceColor','w','color',Deepblue);
plot(kernel_target_nums, mean_MMS_UCB,'o-', 'Linewidth',lw, 'MarkerFaceColor','w','color',Vio);

% errorbar(kernel_target_nums, mean_MMS_DGram,std_MMS_DGram,'Linewidth',lw,'color',Deepred)
% errorbar(kernel_target_nums, mean_MMS_ConEN,std_MMS_ConEN,'Linewidth',lw,'color',Deepgreen)
% errorbar(kernel_target_nums, mean_MMS_Error,std_MMS_Error,'Linewidth',lw,'color',Deepblue)
% errorbar(kernel_target_nums, mean_MMS_UCB,std_MMS_UCB,'Linewidth',lw,'color',Vio)

legend('GSES','VBS','EBS','UCB-BS');
hold off;

% figure;s
subplot(5,2,2);
ba =boxchart(MUS_DGram);
ba.BoxFaceColor = Deepred;

subplot(5,2,4);
bb =boxchart(MUS_ConEN);
bb.BoxFaceColor = Deepgreen;

subplot(5,2,6);
bc = boxchart(MUS_Error);
bc.BoxFaceColor = Deepblue;

subplot(5,2,8);
bd = boxchart(MUS_UCB);
bd.BoxFaceColor = Vio;
% grid on;
% h = gca;
% set(h,'LineWidth',1,'GridLineStyle','--','GridAlpha',1,'GridColor',[0.92,0.9,0.88])
subplot(5,2,10);
hold on;
plot(kernel_target_nums, mean_MUS_DGram,'o-', 'Linewidth',lw, 'MarkerFaceColor','w','color',Deepred);
plot(kernel_target_nums, mean_MUS_ConEN,'o-', 'Linewidth',lw, 'MarkerFaceColor','w','color',Deepgreen);
plot(kernel_target_nums, mean_MUS_Error,'o-', 'Linewidth',lw, 'MarkerFaceColor','w','color',Deepblue);
plot(kernel_target_nums, mean_MUS_UCB,'o-', 'Linewidth',lw, 'MarkerFaceColor','w','color',Vio);
legend('GSES','VBS','EBS','UCB-BS');
hold off;
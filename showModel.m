load('show1000.mat')

Deepred = '#b4534b';
Peach   = '#f58f98';

Deepgreen = '#005831';
Banboo = '#72baa7';

Ethereal = '#2468a2';
Forget_me_not = '#7bbfea';

WGray = '#464547';


time = linspace(0,20,20000);
% R_LineWidth = 0.5;
E_LineWidth = 0.7;
M_LineWidth = 1;

LineColor = {Deepred, Deepgreen, Ethereal, Deepred, Deepgreen, Ethereal};
PatchColor = {Peach, Banboo, Forget_me_not, Peach, Banboo, Forget_me_not};

set(gcf,'Position',[200,100,400,600]);
for i = 1:6
    subplot(6,1,i)
    hold on;
    s1 = shadedErrorBar(time, predict_y_s(i,:), uncertainty, 'lineprops', '-');
    set(s1.edge,'LineWidth',E_LineWidth,'LineStyle',':','Color',LineColor{i});
    s1.mainLine.LineWidth = M_LineWidth;
    s1.mainLine.Color = LineColor{i};
    s1.patch.FaceColor = PatchColor{i};
    plot(time,true_y_s(i,:), 'color',WGray,'LineWidth',E_LineWidth);
%     legend('Estimated','Real');
    hold off;
end

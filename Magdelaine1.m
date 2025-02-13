function Magdelaine1(pat)

% glucose-insulin-carbohydrates 5-ODE model from Magdelaine et al
% Marta, 21/03/2024

disp(['Running patient ' num2str(pat) '...'])

% model parameters
if pat==2
    %  IF2
    M = 72;   % patient mass (kg)
    ksi = 197; % Insulin sensitivity (mg glucose/U/min)
    kl = 1.94;    % Liver endogenous glucose consumption (mg glucose/dL/min)
    Tu = 122; % min
    ku_Vi = 59e-3; % min/dl
    Tr = 183; % min
    kr_Vb = 2.4e-3; % min/dl
    tend=48*60;
elseif pat==3
    % IF3
    M = 94;   % patient mass (kg)
    ksi =  274; % Insulin sensitivity (mg glucose/U/min)
    kl = 1.72;    % Liver endogenous glucose consumption (mg glucose/dL/min)
    Tu = 88; % min
    ku_Vi = 62e-3; % min/dl
    Tr = 49; % min
    kr_Vb = 2e-3; % min/dl
    tend=24*60;
else
    error('We do not know that patient.')
end
% model parameters
Vb = 0.65*M; % dL blood
Vi = 2.5*M; % dL for insulin
kb = 128/M;   % Brain endogenous glucose consumption (mg glucose/dL/min)


% timings (all in min)
dt=0.1; 
t=dt:dt:tend;
gathert=1/dt;
tsave=dt:1:tend;

A=[0 -ksi 0 1 0; 0 0 1 0 0; 0 -Tu^(-2) -2/Tu 0 0; ...
    0 0 0 0 1; 0 0 0 -Tr^(-2) -2/Tr];
B=[0 0; 0 0; ku_Vi/Tu^2 0; 0 0; 0 kr_Vb/Tr^2];
E = [kl-kb; 0; 0; 0; 0];


Ieq = (kl-kb)/ksi;
if pat==2
    X0=[200; Ieq; 0; 0; 0]; % IF2
elseif pat ==3
    X0=[125; Ieq; 0; 0; 0]; % IF3
end
X= X0;


% Input carbohydrates (mg) 
rt=zeros(size(t));

if pat==2
    rt(t==24*60)=128;
    rt(t==25.5*60)=15;
    rt(t==37*60)=150;
    rt(t==41*60)=100;
    rt(t==42.5*60)=7.5;
    rt(t==44.5*60)=15;

% IF2 Thomas (starts at 24h)
% rt(t==7.5*60)=128;
% rt(t==9*60)=15;
% rt(t==20.5*60)=150;

elseif pat==3
% IF3
    rt(t==3*60)=15;
    rt(t==6*60)=15;
    rt(t==7*60)=20;
    rt(t==12.5*60)=15;
    rt(t==14*60)=15;
end

rt=rt*1000;

% Input insulin (U)

% IF2 - Thomas (starts at 24h)
% ut=ones(size(t))/(60*dt);
% ut(t==7.5*60)=ut(t==7.5*60)+2;
% ut(t==9*60)=ut(t==7.5*60)+21.5;
% ut(t==20.5*60)=ut(t==20.5*60)+17.3;
% ut(t==21*60)=ut(t==21*60)+17;

ut=ones(size(t))/60;
if pat==2
    % IF2 - Marta (starts at 0h)
    ut(t>=60&t<=11*60)=0.8/60;
    ut(t==7.5*60)=ut(t==7.5*60)+0.5;
    ut(t==12.5*60)=ut(t==12.5*60)+2;
    ut(t==17*60)=ut(t==17*60)+2;
    ut(t==24*60)=ut(t==24*60)+21.5;
    ut(t==37*60)=ut(t==37*60)+17.3;
    ut(t==37.5*60)=ut(t==37.5*60)+17;
    ut(t==42.5*60)=ut(t==42.5*60)+16;

elseif pat==3
    % IF3
    ut=ones(size(t))*2/60;
    ut(t>=4*60&t<8*60)=1.5/60;
end

% Forward Euler solver
j=1;
Xsave=zeros(size(tsave,1),5);
for i=1:length(t)
    col=[ut(i) rt(i)]';
    dXdt = A*X + B*col + E;
    X = X + dXdt*dt;
    if ~mod(i,gathert)
        Xsave(j,:)=X;
        j=j+1;
    end
end

% plotting
var={'Glucose','Insulin','Insulin Rate',...
    'Digestion','Digestion Rate'};
labe={'U/dL/min','mg/dL/min','mg/dL/min^2'};
for i=1:5
    subplot(3,2,i)
    if i==1
        plot(tsave/60,Xsave(:,i),'LineWidth',2)
        hold all
        plot(t/60,rt/1000,'LineWidth',2)
        title(['Patient ' num2str(pat)],'FontSize',18)
        legend('G (mg/dL)','CHO (g)','Location','Best')
    elseif i==2
        semilogy(tsave/60,Xsave(:,i),'LineWidth',2)
        hold all
        semilogy(t/60,ut,'LineWidth',2)
        legend('I (U/dL)','Iext (U)','Location','Best')
    else
        plot(tsave/60,Xsave(:,i),'LineWidth',2)
        legend(labe{i-2},'Location','Best')
    end
    grid on
    set(gca,'FontSize',14)
    xlabel('Time (h)')
    ylabel(var{i})
end

% finish and save outputs to same folder
disp('Done.')
% gtext(['Patient ' num2str(pat)],'FontSize',20)
save(['Pat' num2str(pat) '.mat'],'-v7')
saveas(gcf,['Pat' num2str(pat) '.png'])
saveas(gcf,['Pat' num2str(pat) '.fig'])
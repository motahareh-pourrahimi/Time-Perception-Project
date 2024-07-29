%% loading data
clear
Num_of_Subjects =17;
for i=1:Num_of_Subjects
    filename=strcat('D:\CNCS\New-Analysis\NewData\','Subject',num2str(i),'Run1.mat');
    temp=load(filename);
    s(i).Estimat=temp.EstimatedTime;
    s(i).Estimat = sortrows(s(i).Estimat);
    %%% Cleaning data
    if(i==15)
        r = temp.EstimatedTime;
        r(9,3) = (r(9,2)+r(9,4))/2;
        s(i).Estimat = r;
    end
    %%%%% claening done
    Taret_Time=repmat(s(i).Estimat(:,1),1,50);
    s(i).ErrorMagnitude=abs(s(i).Estimat(:,2:51)*1000-Taret_Time);
    s(i).MeanErr=mean(s(i).ErrorMagnitude,2);
    s(i).MeanErr=[s(i).Estimat(:,1) s(i).MeanErr];
    s(i).X=s(i).MeanErr(:,1);
    s(i).Y=s(i).MeanErr(:,2);
    s(i).Estimat = (s(i).Estimat(:,2:51)).*1000;
end

run1=s;



%%%%%
clear s;
Num_of_Subjects=17;
%%loading data
for i=1:Num_of_Subjects
    filename=strcat('D:\CNCS\New-Analysis\NewData\','Subject',num2str(i),'Run2.mat');
    temp=load(filename);
    s(i).Estimat=temp.EstimatedTime;
    s(i).Estimat = sortrows(s(i).Estimat);
    Target_Time=repmat(s(i).Estimat(:,1),1,50);
    s(i).ErrorMagnitude=abs(s(i).Estimat(:,2:51).*1000-Target_Time);
    s(i).MeanErr=mean(s(i).ErrorMagnitude,2);
    s(i).MeanErr=[s(i).Estimat(:,1) s(i).MeanErr];
    s(i).X=s(i).MeanErr(:,1);
    s(i).Y=s(i).MeanErr(:,2);
    s(i).Estimat = (s(i).Estimat(:,2:51)).*1000;
end
run2=s;
clear s
clear i
clear temp
%% Fig 2-B mean estimated time vs. target time grouped data, linear regression should be added
all_estimated_time = zeros(10,17);
all_estimated_time2 = zeros(10,17);
for i = 1:Num_of_Subjects
    estimated_time = run1(i).Estimat(:,21:50);
    all_estimated_time(:,i) = mean(estimated_time,2);
    estimated_time2 = run2(i).Estimat(:,21:50);
    all_estimated_time2(:,i) = mean(estimated_time2,2);
end
Mean1 = mean(all_estimated_time,2);
Mean2 = mean(all_estimated_time2,2);
sem1=std(all_estimated_time,0,2)./sqrt(Num_of_Subjects);
sem2=std(all_estimated_time2,0,2)./sqrt(Num_of_Subjects);

[p1,s1] = polyfit(Target_Time(:,1),Mean1,1);
y1 = polyval(p1,Target_Time(:,1));
[p2,s2] = polyfit(Target_Time(:,1),Mean2,1);
y2 = polyval(p2,Target_Time(:,1));

[R1,P1]=corrcoef(Mean1,y1);
pp1 = P1(1,2);
rr1 = R1(1,2);
[R2,P2]=corrcoef(Mean2,y2);
pp2 = P2(1,2);
rr2 = R2(1,2);

figure
plot(Target_Time(:,1),y1);
hold on
plot(Target_Time(:,1),y2);
hold on
scatter(Target_Time(:,1),Mean1,'filled','blue');
hold on
errorbar(Target_Time(:,1),Mean1,sem1,'blue','LineStyle','none');
hold on
text(700,1000,['R^2 = ',num2str(rr1),',p = ',num2str(pp1)]);
hold on
scatter(Target_Time(:,1),Mean2,'filled','r');
hold on
errorbar(Target_Time(:,1),Mean2,sem2,'r','LineStyle','none');
hold on
text(700,950,['R^2 = ',num2str(rr2),',p = ',num2str(pp2)]);
legend('Session 1','Session 2');
title ("Grouped Data");
xlabel('Target Time');
ylabel('Mean produced time');
%% Fig 2-A mean of produced time vs. target time individual subjects, sample = 13th subject
targets=550:50:1000;
target=targets';
het_index = zeros(17,2);
% num = 1
for i=1:17
    if i==13

        continue;
    end
    % subplot(4,4,num)
    figure
    % num = num+1
   data_1= run1(i).Estimat;
    data_2=run2(i).Estimat;
    m1=mean(data_1,2);
    m2=mean(data_2,2);
    sem1=std(data_1,0,2)./sqrt(50);
    sem2=std(data_2,0,2)./sqrt(50);
    
    [p1,s1] = polyfit(Target_Time(:,1),m1,1);
    y1 = polyval(p1,Target_Time(:,1));
    [p2,s2] = polyfit(Target_Time(:,1),m2,1);
    y2 = polyval(p2,Target_Time(:,1));
    
    
    
    [R1,P1]=corrcoef(m1,y1);
    pp1 = P1(1,2);
    rr1 = R1(1,2);
    [R2,P2]=corrcoef(m2,y2);
    pp2 = P2(1,2);
    rr2 = R2(1,2);
    het_index(i,1) = 1-rr1;
    het_index(i,2) = 1-rr2;
    
    %%% Fig 2 - A modification box plot
    data_1_to_box = data_1';
    data_2_to_box = data_2';


    boxplot(data_1_to_box);
    hold on
    boxplot(data_2_to_box);
    hold on
    SIZE = length(data_1_to_box(:,1));
    % .*(1+(rand(SIZE,1)-0.8)/10
    for kk=1:10
        scatter(kk*ones(SIZE,1),data_1_to_box(1:end,kk),50,'blue','filled','MarkerFaceAlpha', 0.6);
        hold on
    end
    hold on
    for kk=1:10
        scatter(kk*ones(SIZE,1),data_2_to_box(1:end,kk), 50,'s','r','filled','MarkerFaceAlpha', 0.6);
        hold on
    end
    hold on
    tt= 1:1:10;
    tt=target;
     

    [ip, ix] = sort(target);
    
    set(gca,'xtick',1:2:10,'xticklabel',num2cell(target),'FontSize', 14);
    hold on
    
 
    scatter(tt,m1,'blue','filled');
    hold on

    if round(pp1,3) == 0
        p_print1 = ', p<0.001';
    else
        p_print1 = [', p=', num2str(round(pp1,3))];
    end
    text(2, 1950,['R^2=',num2str(round(rr1,3)), p_print1]);

    hold on

    if round(pp2,3) == 0
        p_print2 = ', p<0.001';
    else
        p_print2 = [', p=', num2str(round(pp2,3))];
    end

    scatter(tt,m2,'red','filled');
    hold on
    text(2, 1850,['R^2=',num2str(round(rr2,3)),p_print2]);

    hold on
    plot(1:10,y1,'blue','LineWidth',2);
    hold on
    plot(1:10,y2,'red','LineWidth',2);
    
    ylim([0 2100]);
    % legend([s1,s2],'\fontsize{8} Session 1','\fontsize{8} Session 2')
    % legend boxoff
    title(['Produced Time Participant ' , num2str(i)])
    xlabel('Target Time (ms)')
    ylabel('Produced Time (ms)')

    saveas(gcf,['S_Fig_1_Sub_', num2str(i), '.pdf'])
    % hold off
    % break
end

% %     figure

% %     for kk=1:10
% %         subplot(1,10,kk)
% %         histogram(data_1_to_box(:,kk));
% %     end
% %
% %     s(1) = subplot(4,1,1:3);
% %     histogram(x)
% % %     yy = 0:1:100;
% % %     plot(med,yy, 'linewidth',2)
% %     grid on
% %     s(2) = subplot(4,1,4);
% %     boxplot(x, 'Orientation','Horizontal')
% %     grid on
% %     linkaxes(s, 'x')
% end
%% Fig 2-C: std vs. target time for individuals , sample : sub 13
residues = zeros(17,2);
targets=550:50:1000;
target=targets';
heterogeneity_index = zeros(17,2);
for sub = 1:17
    if sub==13
        continue;
    end
    data_1= run1(sub).Estimat;
    data_2=run2(sub).Estimat;
    % sub = 13;
    m1 = std(run1(sub).Estimat,0,2);
    m2 = std(run2(sub).Estimat,0,2);
    
    
    [p1,s1] = polyfit(Target_Time(:,1),m1,1);
    y1 = polyval(p1,Target_Time(:,1));
    [p2,s2] = polyfit(Target_Time(:,1),m2,1);
    y2 = polyval(p2,Target_Time(:,1));
    
    residues(sub,1) = mean( abs(m1-y1) );
    residues(sub,2) = mean( abs(m2-y2) );
    
    [R1,P1]=corrcoef(m1,y1);
    pp1 = P1(1,2);
    rr1 = R1(1,2);
    [R2,P2]=corrcoef(m2,y2);
    pp2 = P2(1,2);
    rr2 = R2(1,2);
    heterogeneity_index(sub , 1) = 1 - rr1;
    heterogeneity_index(sub , 2) = 1 - rr2;
    
    
    
    boot_HIs_1 = zeros(1,1000);
    boot_HIs_2 = zeros(1,1000);
    for i=1:1000
        bootstrapped_1 = datasample(data_1',50);
        bootstrapped_2 = datasample(data_2',50);
        
        m1_boot= std(bootstrapped_1,0,1)';
        m2_boot= std(bootstrapped_2,0,1)';
        [p1,s1] = polyfit(Target_Time(:,1),m1_boot,1);
        y1 = polyval(p1,Target_Time(:,1));
        [p2,s2] = polyfit(Target_Time(:,1),m2_boot,1);
        y2 = polyval(p2,Target_Time(:,1));
        [R1,P1]=corrcoef(m1_boot,y1);
        pp1 = P1(1,2);
        rr1 = R1(1,2);
        [R2,P2]=corrcoef(m2_boot,y2);
        pp2 = P2(1,2);
        rr2 = R2(1,2);
        
        boot_HIs_1(i) = 1 - rr1;
        boot_HIs_2(i) = 1 - rr2;
    end
    sem_HIs_1= std(boot_HIs_1,0,2);
    sem_HIs_2= std(boot_HIs_2,0,2);
    mean_boot_HIs_1 = mean(boot_HIs_1);
    mean_boot_HIs_2 = mean(boot_HIs_2);
    
    
    %%% Fig 2 - C modification box plot
    boot_stds_1 = zeros(10,1000);
    boot_stds_2 = zeros(10,1000);
    for i=1:1000
        bootstrapped_1 = datasample(data_1',50);
        bootstrapped_2 = datasample(data_2',50);
        boot_stds_1(:,i) = std(bootstrapped_1,0,1)';
        boot_stds_2(:,i) = std(bootstrapped_2,0,1)';
    end
    sem_stds_1= std(boot_stds_1,0,2);%./1000;
    sem_stds_2= std(boot_stds_2,0,2);%./1000;
    mean_boot_stds_1 = mean(boot_stds_1,2);
    mean_boot_stds_2 = mean(boot_stds_2,2);
    
    figure
    all_stds_1 = std(run1(sub).Estimat,0,2);
    all_stds_2 = std(run2(sub).Estimat,0,2);
    % subplot(2,1,1)
        plot(550:50:1000,mean_boot_stds_1); % all_stds_1
    % plot(Target_Time(:,1),y1);
    hold on
        plot(550:50:1000,mean_boot_stds_2); 
    % plot(Target_Time(:,1),y2);
    set(gca,'xtick',550:100:1000,'xticklabel',num2cell(target),'FontSize', 14);
    hold on
    scatter(550:50:1000,mean_boot_stds_1,'filled','blue');
    hold on
    errorbar(Target_Time(:,1),mean_boot_stds_1,sem_stds_1,'blue','LineStyle','none');
    hold on
    scatter(550:50:1000,mean_boot_stds_2,'filled','r');
    
    
    
    hold on
    errorbar(Target_Time(:,1),mean_boot_stds_2,sem_stds_2,'r','LineStyle','none');
    legend('Session 1','Session2 ');
    legend boxoff
    ylabel('Temporal Variability (ms)');
    xlabel('Target Time (ms)');
    title(['Temporal Variability Participant ' , num2str(sub)]);
    % 
    % data_1_to_box = data_1';
    % data_2_to_box = data_2';
    % subplot(2,1,2)
    % boxplot(data_1_to_box);
    % hold on
    % boxplot(data_2_to_box);
    % hold on
    % SIZE = length(data_1_to_box(:,1));
    % % .*(1+(rand(SIZE,1)-0.8)/10
    % for kk=1:10
    %     scatter(kk*ones(SIZE,1),data_1_to_box(:,kk),'blue','filled');
    %     hold on
    % end
    % hold on
    % for kk=1:10
    %     scatter(kk*ones(SIZE,1),data_2_to_box(:,kk),'s','r','filled');
    %     hold on
    % end
    % hold on
    % tt= 1:1:10;
    % plot(tt,y1,'blue','LineWidth',2);
    % hold on
    % plot(tt,y2,'r','LineWidth',2);
    % hold on
    % scatter(tt,m1,'blue','filled');
    % hold on
    % text(700,950,['R^2 = ',num2str(rr1),',p = ',num2str(pp1)]);
    % hold on
    % scatter(tt,m2,'s','r','filled');
    % hold on
    % text(700,900,['R^2 = ',num2str(rr2),',p = ',num2str(pp2)]);
    % legend('Session 1','Session 2');
    % title('Produced Time Sample Participant')
    % xlabel('Target Time (ms)')
    % ylabel('Produced Time (ms)')
    % end
    %%% proof of significancy of the heterogeneous pattern
    %%% you should run this code both for run1, and run2
    hold off
    saveas(gcf,['S_Fig_2_Sub_', num2str(sub), '.pdf'])

end

% p=zeros(10,10);
% for i=1:10
%     for j=1:10
%         [~,p(i,j)] = vartest2(run2(sub).Estimat(i,:)',run2(sub).Estimat(j,:)');
%         %close all
%     end
% end
% % h = p<0.2;  %% one tailed bezanam 0.2/2 = 0.1
% h2=p<0.1; %% one- tailed nakonim
% 
%%
ps=zeros(17,9);
stats=zeros(17,9);
for sub=1:17
    for j=1:9
        [h,ps(sub,j),ci,s] = vartest2(run1(sub).Estimat(j,:), run1(sub).Estimat(j+1,:));
         stats(sub,j) = s.fstat;
    end
end
ps
stats





%% Figure 2-D : mean error magnitude vs. target time, subject 13
HI_errror = zeros(17,2);
for sub=1:17
    if sub==13
        continue;
    end
    vec = 550:50:1000;
    target = vec';
    d1=run1(sub).Estimat;
    d2=run2(sub).Estimat;
    dif1=abs(d1-target);
    dif2=abs(d2-target);
    m1=mean(dif1,2);
    m2=mean(dif2,2);
    sem1=std(dif1,0,2)./sqrt(50);
    sem2=std(dif2,0,2)./sqrt(50);
    
    [p1,s1] = polyfit(Target_Time(:,1),m1,1);
    y1 = polyval(p1,Target_Time(:,1));
    [p2,s2] = polyfit(Target_Time(:,1),m2,1);
    y2 = polyval(p2,Target_Time(:,1));
    
    [R1,P1]=corrcoef(m1,y1);
    pp1 = P1(1,2);
    rr1 = R1(1,2);
    [R2,P2]=corrcoef(m2,y2);
    pp2 = P2(1,2);
    rr2 = R2(1,2);
    HI_errror(sub , 1) = 1 - rr1;
    HI_errror(sub , 2) = 1 - rr2;
    
    boot_HIs_1 = zeros(1,1000);
    boot_HIs_2 = zeros(1,1000);
    for i=1:1000
        bootstrapped_dif_1 = datasample(dif1',50);
        bootstrapped_dif_2 = datasample(dif2',50);
        
        m1_boot=mean(bootstrapped_dif_1,1)';
        m2_boot=mean(bootstrapped_dif_2,1)';
        [p1,s1] = polyfit(Target_Time(:,1),m1_boot,1);
        y1 = polyval(p1,Target_Time(:,1));
        [p2,s2] = polyfit(Target_Time(:,1),m2_boot,1);
        y2 = polyval(p2,Target_Time(:,1));
        [R1,P1]=corrcoef(m1_boot,y1);
        pp1 = P1(1,2);
        rr1 = R1(1,2);
        [R2,P2]=corrcoef(m2_boot,y2);
        pp2 = P2(1,2);
        rr2 = R2(1,2);
        
        boot_HIs_1(i) = 1 - rr1;
        boot_HIs_2(i) = 1 - rr2;
    end
    sem_HIs_1= std(boot_HIs_1,0,2);
    sem_HIs_2= std(boot_HIs_2,0,2);
    mean_boot_HIs_1 = mean(boot_HIs_1);
    mean_boot_HIs_2 = mean(boot_HIs_2);
    
    figure
    plot(550:50:1000,m1); %m1
    set(gca,'xtick',550:100:1000,'xticklabel',num2cell(target),'FontSize', 14);
    hold on
    plot(550:50:1000,m2); %m2
    hold on
    scatter(550:50:1000,m1,'filled','blue');%m1
    hold on
    errorbar(Target_Time(:,1),m1,sem1,'blue','LineStyle','none');%m1 , sem1
    hold on
    scatter(550:50:1000,m2,'filled','r');
    hold on
    errorbar(Target_Time(:,1),m2,sem2,'r','LineStyle','none');
    
    % legend('Session 1','Session 2');
    xlabel('Target Time (ms)');
    ylabel('Mean Error Magnitude (ms)');
    title(['Temporal Error Participant ' , num2str(sub)]);
    hold off
    saveas(gcf,['S_Fig_3_Sub_', num2str(sub), '.pdf'])
end
%% Figure 3.A, B : MDS on STD vs. Target time
%%% should run it for both mean error magnitude and stds --> all 4 pics of
%%% figure 3

target=550:50:1000;
t=target';
stds_1= zeros(17,10);
for i=1:17
    stds_1(i,:) = mean(abs(run1(i).Estimat(:,2:end) - t),2)';%%% just name of variable is stds_1 , it is actually mean error magitude_1
%             stds_1(i,:)=std(run1(i).Estimat,0,2)';
end
stds_2= zeros(17,10);
for i=1:17
    stds_2(i,:) = mean(abs(run2(i).Estimat(:,2:end) - t),2)';%%% just name of variable is stds_2 , it is actually mean error magitude_2
%             stds_2(i,:)=std(run2(i).Estimat,0,2)';
end

correct_D = pdist([stds_1;stds_2],'Euclidean');
[correct_A] = cmdscale(correct_D,2);

correct_XY = [correct_A(:,1),correct_A(:,2)];

realDist = diag( pdist2(correct_A(1:17,:),correct_A(18:end,:),'Euclidean') );

% drawing MDS 2-D graph
%%% 17 distinguishable colors
color(1).c = [128/256,0,0];
color(2).c = [170/256,110/256,40/256];
color(3).c = [128/256,128/256,0];
color(4).c = [0,128/256,128/256];
color(5).c = [0,0,128/256];
color(6).c = [0,0,0];
color(7).c = [128,128,128]./256;
color(8).c = [240,50,230]./256;
color(9).c = [70,240,240]./256;
color(10).c = [210,245,60]./256;
color(11).c = [255,255,25]./256;
color(12).c = [230,25,75]./256;
color(13).c = [170,255,195]./256;
color(14).c = [245,130,48]./256;
color(15).c = [250,190,212]./256;
color(16).c = [60,180,75]./256;
color(17).c = [145,30,180]./256;


figure(1)
for i=1:17
    marker_color=color(i).c;
    scatter(correct_A(i,1) , correct_A(i,2),100,marker_color,'filled');
    %text(correct_A(i,1) , correct_A(i,2),num2str(i),'Color',marker_color,'FontSize',10);
    hold on
end

%%%plot 2-D to see if run2 is close to run1

for i=18:34
    marker_color=color(i-17).c;
    hold on
    scatter(correct_A(i,1) , correct_A(i,2),100,marker_color,'filled','s');
    %sz=60;
    %text(A2(i,1) , A2(i,2),['R',num2str(i)],'Color',marker_color,'FontSize',10);
    hold on
end
xlabel('Dimension (axis) 1');
ylabel('Dimention (axis) 2');
title('MDS on Mean Error MAgnitude '); %standard deviations

%%%  randomized hypothesis testing : Main feature space NOT the 2-D space


realDist = diag( pdist2(stds_1,stds_2,'Euclidean') );

AllDistances=zeros(17,1);
XY = stds_1;
XY2 = stds_2;
for i=1:10000
    shuffledXY = XY(randperm(size(XY,1)),:);
    shuffledXY2 = XY2(randperm(size(XY2,1)),:);
    distance=diag( pdist2(shuffledXY , shuffledXY2,'Euclidean') );
    AllDistances=[AllDistances,distance];
end
AllDistances=AllDistances(:,2:end);

MeanDist=mean(AllDistances);
MeanRealDistances=mean(realDist);

figure
histogram(MeanDist);

x=mean(MeanDist,2);
yf = 550; %% 230 for stds
y=0:1:yf;
line([x x],[y(1) y(end)-8],'LineWidth',2.5,'Color','cyan');
hold on
text(x -0.01, y(end)-4, "Mean", 'Color','cyan','FontSize',10);

hold on

x=MeanRealDistances;
y=0:1:yf;
line([x x],[y(1) y(end)-8],'LineWidth',2.5,'Color','r');
hold on
text(x-0.02, y(end)-4, "RealMean", 'Color','r','FontSize',10);
title("Histogram of Mean Distances of MDS with Shuffled Labels");
ylabel('Frequency','FontSize',10);
xlabel("mean distance between run 1 and run 2 in each shuffling ",'FontSize',10);

%%% plot the cdf
figure
ecdf(MeanDist);
title("cdf");
%%% CDF & p_value
[f,x]=ecdf(MeanDist);
index =find( (x == MeanRealDistances )| (x>MeanRealDistances) , 1 );
p_value = f(index);

Consistency_measure = abs(MeanRealDistances - mean(MeanDist,2)) / std(MeanDist,0,2);

%% RDM
similarity_of_two_runs = zeros(2,17);

for i=1:17
    data = run1(i).Estimat(:,2:50)';
    [R,P] = corr(data);  % ,'type','Kendall'
    RDM = 1 - R;
    % figure
    % 
    % imagesc(RDM);
    % % set(gca,'xtick',550:100:1000,'xticklabel',num2cell(target),'FontSize', 14);
    % colormap('jet');
    % colorbar ;
    %  % title(['RDM Sample Participant Session 1']);
    % title([ 'Participant ',num2str(i), ' Session 1']);
    % saveas(gcf,['S_Fig_4_Sub_', num2str(i), '_Session_1','.pdf'])
    data2 = run2(i).Estimat(:,2:50)';
    
    [R2,P2] = corr(data2); % ,'type','Kendall'
    RDM2 = 1 - R2;
    % figure
    % imagesc(RDM2);
    % % set(gca,'xtick',550:100:1000,'xticklabel',num2cell(target),'FontSize', 14);
    % colormap('jet');
    % colorbar ;
    % % title(['RDM Sample Participant Session 2']);
    % title([Participant ',num2str(i), ' Session 2']);
    % saveas(gcf,['S_Fig_4_Sub_', num2str(i), '_Session_2','.pdf'])
    %%% Comaprison of RDMs
    %%% second order dissimilarity  : Comaprison of RDMs
    [Rf,Pf] = corrcoef(RDM ,RDM2);
    similarity_of_two_runs(1,i) = Rf(1,2);
    similarity_of_two_runs(2,i) = Pf(1,2);
    % close all
    % break
end 

similarity_of_two_runs
%%% Comaprison of RDMs
%% New Learning individuals

T = zeros(10,49,17);
T2 = zeros(10,49,17);

for sub = 1:17
    temp = run1(sub).Estimat;%run1(sub).ErrorMagnitude(:,2:end);
    T(:,:,sub) = temp(:,2:50);
    temp2 = run2(sub).Estimat;%run2(sub).ErrorMagnitude(:,2:end);
    T2(:,:,sub) = temp2(:,2:50);
end

target=550:50:1000;
t=target';
dimension_3_lentgh = 17;
num_of_rows = 10;
num_of_features = 49;
results = zeros(dimension_3_lentgh,2);
%for subject = 1:dimension_3_lentgh
     subject = 13;
    stds_1 = T(:,:,subject);
    stds_2 = T2(:,:,subject);
    
    correct_D = pdist([stds_1;stds_2],'Euclidean');
    [correct_A] = cmdscale(correct_D,2);
    
    correct_XY = [correct_A(:,1),correct_A(:,2)];
    
    realDist = diag( pdist2(stds_1,stds_2,'Euclidean') );
    
    % drawing MDS 2-D graph
    %%% 17 distinguishable colors
    color(1).c = [128/256,0,0];
    color(2).c = [170/256,110/256,40/256];
    color(3).c = [128/256,128/256,0];
    color(4).c = [0,128/256,128/256];
    color(5).c = [0,0,128/256];
    color(6).c = [0,0,0];
    color(7).c = [128,128,128]./256;
    color(8).c = [240,50,230]./256;
    color(9).c = [70,240,240]./256;
    color(10).c = [210,245,60]./256;
    color(11).c = [255,255,25]./256;
    color(12).c = [230,25,75]./256;
    color(13).c = [170,295,175]./256;
    color(14).c = [245,130,48]./256;
    color(15).c = [250,190,212]./256;
    color(16).c = [60,180,75]./256;
    color(17).c = [145,30,180]./256;
    
    
    figure(1)
    for i=1:num_of_rows
        marker_color=color(i).c;
        scatter(correct_A(i,1) , correct_A(i,2),100,marker_color,'filled');
        %text(correct_A(i,1) , correct_A(i,2),num2str(i),'Color',marker_color,'FontSize',10);
        hold on
    end
    xlabel('Dimension (axis) 1');
    ylabel('Dimention (axis) 2');
    title('MDS on Mean Error Magnitude');
    
    
    %%%plot 2-D to see if run2 is close to run1
    
    for i=num_of_rows+1:2*num_of_rows
        marker_color=color(i-num_of_rows).c;
        hold on
        scatter(correct_A(i,1) , correct_A(i,2),100,marker_color,'filled','s');
        %sz=60;
        %text(A2(i,1) , A2(i,2),['R',num2str(i)],'Color',marker_color,'FontSize',10);
        hold on
    end
    xlabel('Dimension (axis) 1');
    ylabel('Dimention (axis) 2');
    title('MDS on temporal learning patterns');
    
    %%%  randomized hypothesis testing : Main feature space NOT the 2-D space
    
    AllDistances=zeros(num_of_rows,1);
    XY = stds_1;
    XY2 = stds_2;
    for i=1:10000
        shuffledXY = XY(randperm(size(XY,1)),:);
        shuffledXY2 = XY2(randperm(size(XY2,1)),:);
        distance=diag( pdist2(shuffledXY , shuffledXY2,'Euclidean') );
        AllDistances=[AllDistances,distance];
    end
    AllDistances=AllDistances(:,2:end);
    
    MeanDist=mean(AllDistances);
    MeanRealDistances=mean(realDist);
    
    figure
    histogram(MeanDist);
    
    x=mean(MeanDist,2);
    yf = 140+600; %% 230 for stds
    y=0:1:yf;
    line([x x],[y(1) y(end)-8],'LineWidth',2.5,'Color','cyan');
    hold on
    text(x -0.01, y(end)-4, "Mean", 'Color','cyan','FontSize',10);
    
    hold on
    
    x=MeanRealDistances;
    y=0:1:yf;
    line([x x],[y(1) y(end)-8],'LineWidth',2.5,'Color','r');
    hold on
    text(x-0.02, y(end)-4, "RealMean", 'Color','r','FontSize',10);
    title("Histogram of Mean Distances of MDS with Shuffled Labels");
    ylabel('Frequency','FontSize',10);
    xlabel("mean distance between run 1 and run 2 in each shuffling ",'FontSize',10);
    
    %%% plot the cdf
    figure
    ecdf(MeanDist);
    title("cdf");
    %%% CDF & p_value
    [f,x]=ecdf(MeanDist);
    index =min( find( (x == MeanRealDistances )| (x>MeanRealDistances) ) );
    p_value = f(index);
    
    Consistency_measure = abs(MeanRealDistances - mean(MeanDist,2)) / std(MeanDist,0,2);
    
    results(subject,1) = Consistency_measure;
    results(subject,2) = p_value;
% end
%% plotting
trials=2:1:50;
heterogeneity_index = zeros(10,2,17);
% for i=1:17
i = 13;
%     figure
    for j=1:10
%         subplot(2,5,j)
        figure
        temp = run1(i).Estimat;
        plot(trials,temp(j,2:end));
        hold on
        temp2 = run2(i).Estimat;
        plot(trials,temp2(j,2:end));
        hold off
        
        [p1,s1] = polyfit(trials,temp(j,2:end),1);
        y1 = polyval(p1,trials);
        [p2,s2] = polyfit(trials,temp2(j,2:end),1);
        y2 = polyval(p2,trials);
        
        [R1,P1]=corrcoef(temp(j,2:end),y1);
        pp1 = P1(1,2);
        rr1 = R1(1,2);
        [R2,P2]=corrcoef(temp2(j,2:end),y2);
        pp2 = P2(1,2);
        rr2 = R2(1,2);
        heterogeneity_index(j , 1,i) = 1 - rr1;
        heterogeneity_index(j , 2,i) = 1 - rr2;
        
        title('Temporal Learning Pattern Sample Participant Sample Target Time');
        xlabel('Trials');
        ylabel('Produced Time (ms)');
        legend('Session 1', 'Session 2');
    end
% end
%% %% RDM for 2 runs
SI = zeros(1,17);
SI_p = zeros(1,17);
for i=1:17
    data = run1(i).ErrorMagnitude';
    data2 = run2(i).ErrorMagnitude';
    [R,P] = corr(data, data2,'type','Kendall'); %,'type','Kendall'
    R
    P
    SI(i) = R;
    SI_p(i) = P;
    RDM = 1 - R;
    figure
    imagesc(RDM);
    colormap('jet');
    colorbar ;
    title(['RDM ' , 'Subject ',num2str(i)]);
end
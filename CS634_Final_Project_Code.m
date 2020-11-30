clc
clear
close all

r = readtable('winequality-red.csv');
w = readtable('winequality-white.csv');

r = table2array(r);
w = table2array(w);

g5 = randperm(340);
r5 = find(r(:,12)==5);
dr5 = r(r5,:);
dr5(g5,:) = [];

g6 = randperm(319);
r6 = find(r(:,12)==6);
dr6 = r(r6,:);
dr6(g6,:) = [];

r_new = [r(find(r(:,12)==3),:);r(find(r(:,12)==4),:);dr5;dr6;r(find(r(:,12)==7),:);r(find(r(:,12)==8),:)];
r_newww = [repmat(r(find(r(:,12)==3),:),10,1);repmat(r(find(r(:,12)==4),:),3,1);dr5;dr6;r(find(r(:,12)==7),:);repmat(r(find(r(:,12)==8),:),7,1)];
r_newww = r_newww(randperm(1244),:);

g5 = randperm(1000);
w5 = find(w(:,12)==5);
dw5 = w(w5,:);
dw5(g5,:) = [];

g6 = randperm(1600);
w6 = find(w(:,12)==6);
dw6 = w(w6,:);
dw6(g6,:) = [];

g7 = randperm(300);
w7 = find(w(:,12)==7);
dw7 = w(w7,:);
dw7(g7,:) = [];

w_new = [w(find(w(:,12)==3),:);w(find(w(:,12)==4),:);dw5;dw6;dw7;w(find(w(:,12)==8),:);w(find(w(:,12)==9),:)];
w_newww = [repmat(w(find(w(:,12)==3),:),6,1);repmat(w(find(w(:,12)==4),:),2,1);dw5;dw6;dw7;repmat(w(find(w(:,12)==8),:),2,1);repmat(w(find(w(:,12)==9),:),20,1)];
w_newww = w_newww(randperm(2531),:);

%%
mean_r_newww = mean(r_newww(:,1:11));
mean_w_newww = mean(w_newww(:,1:11));

std_r_newww = std(r_newww(:,1:11));
std_w_newww = std(w_newww(:,1:11));

for i = 1:11
   r_newww(:,i)=(r_newww(:,i)-mean_r_newww(i))/std_r_newww(i); 
   w_newww(:,i)=(w_newww(:,i)-mean_w_newww(i))/std_w_newww(i); 
end

fr_newww = zeros(1,10);
fw_newww = zeros(1,10);

for i =1:10
   fr_newww(i) = length(find(r_newww(:,12)==i));
   fw_newww(i) = length(find(w_newww(:,12)==i));
end

%% 10-fold cross validation for red-wine SVM
k_size = round(size(r_newww,1)/10);

L_all = cell(10,1);
L_all_rates = cell(10,1);
Prec_all = cell(10,1);
f1 = cell(10,1);
acc = cell(10,1);
err = cell(10,1);
tss = cell(10,1);
hss = cell(10,1);

for i = 0:9
   
    test_r = r_newww(i*k_size+1:(i+1)*k_size,:);
    if i == 0
        train_r = r_newww((i+1)*k_size+1:end,:);
    elseif i == 9
        train_r = r_newww(1:i*k_size,:);
    else
        train_r = [r_newww(1:i*k_size,:);r_newww((i+1)*k_size:end,:)];
    end
    
    SVMModel = fitcecoc(train_r(:,1:11),train_r(:,12));
    p_svm = predict(SVMModel,test_r(:,1:11));
    
    % For 3
    idx_re_3 = find(test_r(:,12) == 3);
    idx_re_not_3 = find(test_r(:,12) ~= 3);
    idx_pr_3 = find(p_svm == 3);
    idx_pr_not_3 = find(p_svm ~= 3);
    L_3 = [length(find(p_svm(idx_re_3)==3)) length(find(p_svm(idx_re_not_3)~=3))...
        length(find(p_svm(idx_re_not_3)==3)) length(find(p_svm(idx_re_3)~=3))];
    L_3_rates = 100*[length(find(p_svm(idx_re_3)==3))/(length(find(p_svm(idx_re_3)==3))+length(find(p_svm(idx_re_3)~=3))) length(find(p_svm(idx_re_not_3)~=3))/(length(find(p_svm(idx_re_not_3)~=3))+length(find(p_svm(idx_re_not_3)==3)))...
        length(find(p_svm(idx_re_not_3)==3))/(length(find(p_svm(idx_re_not_3)==3))+length(find(p_svm(idx_re_not_3)~=3))) length(find(p_svm(idx_re_3)~=3))/(length(find(p_svm(idx_re_3)~=3))+length(find(p_svm(idx_re_3)==3)))];
    prec_3 = length(find(p_svm(idx_re_3)==3))/(length(find(p_svm(idx_re_3)==3))+length(find(p_svm(idx_re_not_3)==3)));
    
    f1_3 = (2*length(find(p_svm(idx_re_3)==3)))/(2*length(find(p_svm(idx_re_3)==3))+length(find(p_svm(idx_re_not_3)==3))+length(find(p_svm(idx_re_3)~=3)));
    
    Tot = length(find(p_svm(idx_re_3)==3)) + length(find(p_svm(idx_re_not_3)~=3)) + ...
        length(find(p_svm(idx_re_not_3)==3)) + length(find(p_svm(idx_re_3)~=3));
    
    acc_3 = (length(find(p_svm(idx_re_3)==3))+length(find(p_svm(idx_re_not_3)~=3)))/Tot;
    
    err_3 = (length(find(p_svm(idx_re_not_3)==3)) + length(find(p_svm(idx_re_3)~=3)))/Tot;
    
    tss_3 = (length(find(p_svm(idx_re_3)==3))/(length(find(p_svm(idx_re_3)==3))+length(find(p_svm(idx_re_3)~=3))))-...
        (length(find(p_svm(idx_re_not_3)==3))/(length(find(p_svm(idx_re_not_3)==3))+length(find(p_svm(idx_re_not_3)~=3))));
    
    % For 4
    idx_re_4 = find(test_r(:,12) == 4);
    idx_re_not_4 = find(test_r(:,12) ~= 4);
    idx_pr_4 = find(p_svm == 4);
    idx_pr_not_4 = find(p_svm ~= 4);
    L_4 = [length(find(p_svm(idx_re_4)==4)) length(find(p_svm(idx_re_not_4)~=4))...
        length(find(p_svm(idx_re_not_4)==4)) length(find(p_svm(idx_re_4)~=4))];
    L_4_rates = 100*[length(find(p_svm(idx_re_4)==4))/(length(find(p_svm(idx_re_4)==4))+length(find(p_svm(idx_re_4)~=4))) length(find(p_svm(idx_re_not_4)~=4))/(length(find(p_svm(idx_re_not_4)~=4))+length(find(p_svm(idx_re_not_4)==4)))...
        length(find(p_svm(idx_re_not_4)==4))/(length(find(p_svm(idx_re_not_4)==4))+length(find(p_svm(idx_re_not_4)~=4))) length(find(p_svm(idx_re_4)~=4))/(length(find(p_svm(idx_re_4)~=4))+length(find(p_svm(idx_re_4)==4)))];
    prec_4 = length(find(p_svm(idx_re_4)==4))/(length(find(p_svm(idx_re_4)==4))+length(find(p_svm(idx_re_not_4)==4)));
    
    f1_4 = (2*length(find(p_svm(idx_re_4)==4)))/(2*length(find(p_svm(idx_re_4)==4))+length(find(p_svm(idx_re_not_4)==4))+length(find(p_svm(idx_re_4)~=4)));
    
    Tot = length(find(p_svm(idx_re_4)==4)) + length(find(p_svm(idx_re_not_4)~=4)) + ...
        length(find(p_svm(idx_re_not_4)==4)) + length(find(p_svm(idx_re_4)~=4));
    
    acc_4 = (length(find(p_svm(idx_re_4)==4))+length(find(p_svm(idx_re_not_4)~=4)))/Tot;
    
    err_4 = (length(find(p_svm(idx_re_not_4)==4)) + length(find(p_svm(idx_re_4)~=4)))/Tot;
    
    tss_4 = (length(find(p_svm(idx_re_4)==4))/(length(find(p_svm(idx_re_4)==4))+length(find(p_svm(idx_re_4)~=4))))-...
        (length(find(p_svm(idx_re_not_4)==4))/(length(find(p_svm(idx_re_not_4)==4))+length(find(p_svm(idx_re_not_4)~=4))));
    
    % For 5
    idx_re_5 = find(test_r(:,12) == 5);
    idx_re_not_5 = find(test_r(:,12) ~= 5);
    idx_pr_5 = find(p_svm == 5);
    idx_pr_not_5 = find(p_svm ~= 5);
    L_5 = [length(find(p_svm(idx_re_5)==5)) length(find(p_svm(idx_re_not_5)~=5))...
        length(find(p_svm(idx_re_not_5)==5)) length(find(p_svm(idx_re_5)~=5))];
    L_5_rates = 100*[length(find(p_svm(idx_re_5)==5))/(length(find(p_svm(idx_re_5)==5))+length(find(p_svm(idx_re_5)~=5))) length(find(p_svm(idx_re_not_5)~=5))/(length(find(p_svm(idx_re_not_5)~=5))+length(find(p_svm(idx_re_not_5)==5)))...
        length(find(p_svm(idx_re_not_5)==5))/(length(find(p_svm(idx_re_not_5)==5))+length(find(p_svm(idx_re_not_5)~=5))) length(find(p_svm(idx_re_5)~=5))/(length(find(p_svm(idx_re_5)~=5))+length(find(p_svm(idx_re_5)==5)))];
    prec_5 = length(find(p_svm(idx_re_5)==5))/(length(find(p_svm(idx_re_5)==5))+length(find(p_svm(idx_re_not_5)==5)));
    
    f1_5 = (2*length(find(p_svm(idx_re_5)==5)))/(2*length(find(p_svm(idx_re_5)==5))+length(find(p_svm(idx_re_not_5)==5))+length(find(p_svm(idx_re_5)~=5)));
    
    Tot = length(find(p_svm(idx_re_5)==5)) + length(find(p_svm(idx_re_not_5)~=5)) + ...
        length(find(p_svm(idx_re_not_5)==5)) + length(find(p_svm(idx_re_5)~=5));
    
    acc_5 = (length(find(p_svm(idx_re_5)==5))+length(find(p_svm(idx_re_not_5)~=5)))/Tot;
    
    err_5 = (length(find(p_svm(idx_re_not_5)==5)) + length(find(p_svm(idx_re_5)~=5)))/Tot;
    
    tss_5 = (length(find(p_svm(idx_re_5)==5))/(length(find(p_svm(idx_re_5)==5))+length(find(p_svm(idx_re_5)~=5))))-...
        (length(find(p_svm(idx_re_not_5)==5))/(length(find(p_svm(idx_re_not_5)==5))+length(find(p_svm(idx_re_not_5)~=5))));
    
    % For 6
    idx_re_6 = find(test_r(:,12) == 6);
    idx_re_not_6 = find(test_r(:,12) ~= 6);
    idx_pr_6 = find(p_svm == 6);
    idx_pr_not_6 = find(p_svm ~= 6);
    L_6 = [length(find(p_svm(idx_re_6)==6)) length(find(p_svm(idx_re_not_6)~=6))...
        length(find(p_svm(idx_re_not_6)==6)) length(find(p_svm(idx_re_6)~=6))];
    L_6_rates = 100*[length(find(p_svm(idx_re_6)==6))/(length(find(p_svm(idx_re_6)==6))+length(find(p_svm(idx_re_6)~=6))) length(find(p_svm(idx_re_not_6)~=6))/(length(find(p_svm(idx_re_not_6)~=6))+length(find(p_svm(idx_re_not_6)==6)))...
        length(find(p_svm(idx_re_not_6)==6))/(length(find(p_svm(idx_re_not_6)==6))+length(find(p_svm(idx_re_not_6)~=6))) length(find(p_svm(idx_re_6)~=6))/(length(find(p_svm(idx_re_6)~=6))+length(find(p_svm(idx_re_6)==6)))];
    prec_6 = length(find(p_svm(idx_re_6)==6))/(length(find(p_svm(idx_re_6)==6))+length(find(p_svm(idx_re_not_6)==6)));
    
    f1_6 = (2*length(find(p_svm(idx_re_6)==6)))/(2*length(find(p_svm(idx_re_6)==6))+length(find(p_svm(idx_re_not_6)==6))+length(find(p_svm(idx_re_6)~=6)));
    
    Tot = length(find(p_svm(idx_re_6)==6)) + length(find(p_svm(idx_re_not_6)~=6)) + ...
        length(find(p_svm(idx_re_not_6)==6)) + length(find(p_svm(idx_re_6)~=6));
    
    acc_6 = (length(find(p_svm(idx_re_6)==6))+length(find(p_svm(idx_re_not_6)~=6)))/Tot;
    
    err_6 = (length(find(p_svm(idx_re_not_6)==6)) + length(find(p_svm(idx_re_6)~=6)))/Tot;
    
    tss_6 = (length(find(p_svm(idx_re_6)==6))/(length(find(p_svm(idx_re_6)==6))+length(find(p_svm(idx_re_6)~=6))))-...
        (length(find(p_svm(idx_re_not_6)==6))/(length(find(p_svm(idx_re_not_6)==6))+length(find(p_svm(idx_re_not_6)~=6))));
    
    % For 7
    idx_re_7 = find(test_r(:,12) == 7);
    idx_re_not_7 = find(test_r(:,12) ~= 7);
    idx_pr_7 = find(p_svm == 7);
    idx_pr_not_7 = find(p_svm ~= 7);
    L_7 = [length(find(p_svm(idx_re_7)==7)) length(find(p_svm(idx_re_not_7)~=7))...
        length(find(p_svm(idx_re_not_7)==7)) length(find(p_svm(idx_re_7)~=7))];
    L_7_rates = 100*[length(find(p_svm(idx_re_7)==7))/(length(find(p_svm(idx_re_7)==7))+length(find(p_svm(idx_re_7)~=7))) length(find(p_svm(idx_re_not_7)~=7))/(length(find(p_svm(idx_re_not_7)~=7))+length(find(p_svm(idx_re_not_7)==7)))...
        length(find(p_svm(idx_re_not_7)==7))/(length(find(p_svm(idx_re_not_7)==7))+length(find(p_svm(idx_re_not_7)~=7))) length(find(p_svm(idx_re_7)~=7))/(length(find(p_svm(idx_re_7)~=7))+length(find(p_svm(idx_re_7)==7)))];
    prec_7 = length(find(p_svm(idx_re_7)==7))/(length(find(p_svm(idx_re_7)==7))+length(find(p_svm(idx_re_not_7)==7)));
    
    f1_7 = (2*length(find(p_svm(idx_re_7)==7)))/(2*length(find(p_svm(idx_re_7)==7))+length(find(p_svm(idx_re_not_7)==7))+length(find(p_svm(idx_re_7)~=7)));
    
    Tot = length(find(p_svm(idx_re_7)==7)) + length(find(p_svm(idx_re_not_7)~=7)) + ...
        length(find(p_svm(idx_re_not_7)==7)) + length(find(p_svm(idx_re_7)~=7));
    
    acc_7 = (length(find(p_svm(idx_re_7)==7))+length(find(p_svm(idx_re_not_7)~=7)))/Tot;
    
    err_7 = (length(find(p_svm(idx_re_not_7)==7)) + length(find(p_svm(idx_re_7)~=7)))/Tot;
    
    tss_7 = (length(find(p_svm(idx_re_7)==7))/(length(find(p_svm(idx_re_7)==7))+length(find(p_svm(idx_re_7)~=7))))-...
        (length(find(p_svm(idx_re_not_7)==7))/(length(find(p_svm(idx_re_not_7)==7))+length(find(p_svm(idx_re_not_7)~=7))));
    
    % For 8
    idx_re_8 = find(test_r(:,12) == 8);
    idx_re_not_8 = find(test_r(:,12) ~= 8);
    idx_pr_8 = find(p_svm == 8);
    idx_pr_not_8 = find(p_svm ~= 8);
    L_8 = [length(find(p_svm(idx_re_8)==8)) length(find(p_svm(idx_re_not_8)~=8))...
        length(find(p_svm(idx_re_not_8)==8)) length(find(p_svm(idx_re_8)~=8))];
    L_8_rates = 100*[length(find(p_svm(idx_re_8)==8))/(length(find(p_svm(idx_re_8)==8))+length(find(p_svm(idx_re_8)~=8))) length(find(p_svm(idx_re_not_8)~=8))/(length(find(p_svm(idx_re_not_8)~=8))+length(find(p_svm(idx_re_not_8)==8)))...
        length(find(p_svm(idx_re_not_8)==8))/(length(find(p_svm(idx_re_not_8)==8))+length(find(p_svm(idx_re_not_8)~=8))) length(find(p_svm(idx_re_8)~=8))/(length(find(p_svm(idx_re_8)~=8))+length(find(p_svm(idx_re_8)==8)))];
    prec_8 = length(find(p_svm(idx_re_8)==8))/(length(find(p_svm(idx_re_8)==8))+length(find(p_svm(idx_re_not_8)==8)));
    
    f1_8 = (2*length(find(p_svm(idx_re_8)==8)))/(2*length(find(p_svm(idx_re_8)==8))+length(find(p_svm(idx_re_not_8)==8))+length(find(p_svm(idx_re_8)~=8)));
    
    Tot = length(find(p_svm(idx_re_8)==8)) + length(find(p_svm(idx_re_not_8)~=8)) + ...
        length(find(p_svm(idx_re_not_8)==8)) + length(find(p_svm(idx_re_8)~=8));
    
    acc_8 = (length(find(p_svm(idx_re_8)==8))+length(find(p_svm(idx_re_not_8)~=8)))/Tot;
    
    err_8 = (length(find(p_svm(idx_re_not_8)==8)) + length(find(p_svm(idx_re_8)~=8)))/Tot;
    
    tss_8 = (length(find(p_svm(idx_re_8)==8))/(length(find(p_svm(idx_re_8)==8))+length(find(p_svm(idx_re_8)~=8))))-...
        (length(find(p_svm(idx_re_not_8)==8))/(length(find(p_svm(idx_re_not_8)==8))+length(find(p_svm(idx_re_not_8)~=8))));
    
    disp('SVM');
    disp('-------------------------');
    disp('-------------------------');
    disp('TP TN FP FN');
    disp('-------------------------');
    L_all{i+1} = [L_3;L_4;L_5;L_6;L_7;L_8];
    cell2mat(L_all)
    disp('TPR TNR FPR FNR');
    disp('-------------------------');
    L_all_rates{i+1} = [L_3_rates;L_4_rates;L_5_rates;L_6_rates;L_7_rates;L_8_rates];
    cell2mat(L_all_rates)
    disp('Precision');
    disp('-------------------------');
    Prec_all{i+1} = [prec_3;prec_4;prec_5;prec_6;prec_7;prec_8];
    cell2mat(Prec_all)
    disp('F1-Score')
    disp('-------------------------');
    f1{i+1} = [f1_3;f1_4;f1_5;f1_6;f1_7;f1_8];
    cell2mat(f1)
    disp('Accuracy')
    disp('-------------------------');
    acc{i+1} = [acc_3;acc_4;acc_5;acc_6;acc_7;acc_8];
    cell2mat(acc)
    disp('Error-rate')
    disp('-------------------------');
    err{i+1} = [err_3;err_4;err_5;err_6;err_7;err_8];
    cell2mat(err)
    disp('TSS')
    disp('-------------------------');
    tss{i+1} = [tss_3;tss_4;tss_5;tss_6;tss_7;tss_8];
    cell2mat(tss)
end


%% 10-fold cross validation for red-wine Naive Bayes

k_size = round(size(r_newww,1)/10);

L_all_nb = cell(10,1);
L_all_rates_nb = cell(10,1);
Prec_all_nb = cell(10,1);
f1_nb = cell(10,1);
acc_nb = cell(10,1);
err_nb = cell(10,1);
tss_nb = cell(10,1);

for i = 0:9
   
    test_r = r_newww(i*k_size+1:(i+1)*k_size,:);
    if i == 0
        train_r = r_newww((i+1)*k_size+1:end,:);
    elseif i == 9
        train_r = r_newww(1:i*k_size,:);
    else
        train_r = [r_newww(1:i*k_size,:);r_newww((i+1)*k_size:end,:)];
    end
    
    NBModel = fitcnb(train_r(:,1:11),train_r(:,12));
    p_nb = predict(NBModel,test_r(:,1:11));
    
    % For 3
    idx_re_3 = find(test_r(:,12) == 3);
    idx_re_not_3 = find(test_r(:,12) ~= 3);
    idx_pr_3 = find(p_nb == 3);
    idx_pr_not_3 = find(p_nb ~= 3);
    L_3 = [length(find(p_nb(idx_re_3)==3)) length(find(p_nb(idx_re_not_3)~=3))...
        length(find(p_nb(idx_re_not_3)==3)) length(find(p_nb(idx_re_3)~=3))];
    L_3_rates = 100*[length(find(p_nb(idx_re_3)==3))/(length(find(p_nb(idx_re_3)==3))+length(find(p_nb(idx_re_3)~=3))) length(find(p_nb(idx_re_not_3)~=3))/(length(find(p_nb(idx_re_not_3)~=3))+length(find(p_nb(idx_re_not_3)==3)))...
        length(find(p_nb(idx_re_not_3)==3))/(length(find(p_nb(idx_re_not_3)==3))+length(find(p_nb(idx_re_not_3)~=3))) length(find(p_nb(idx_re_3)~=3))/(length(find(p_nb(idx_re_3)~=3))+length(find(p_nb(idx_re_3)==3)))];
    prec_3 = length(find(p_nb(idx_re_3)==3))/(length(find(p_nb(idx_re_3)==3))+length(find(p_nb(idx_re_not_3)==3)));
    
    f1_3 = (2*length(find(p_nb(idx_re_3)==3)))/(2*length(find(p_nb(idx_re_3)==3))+length(find(p_nb(idx_re_not_3)==3))+length(find(p_nb(idx_re_3)~=3)));
    
    Tot = length(find(p_nb(idx_re_3)==3)) + length(find(p_nb(idx_re_not_3)~=3)) + ...
        length(find(p_nb(idx_re_not_3)==3)) + length(find(p_nb(idx_re_3)~=3));
    
    acc_3 = (length(find(p_nb(idx_re_3)==3))+length(find(p_nb(idx_re_not_3)~=3)))/Tot;
    
    err_3 = (length(find(p_nb(idx_re_not_3)==3)) + length(find(p_nb(idx_re_3)~=3)))/Tot;
    
    tss_3 = (length(find(p_nb(idx_re_3)==3))/(length(find(p_nb(idx_re_3)==3))+length(find(p_nb(idx_re_3)~=3))))-...
        (length(find(p_nb(idx_re_not_3)==3))/(length(find(p_nb(idx_re_not_3)==3))+length(find(p_nb(idx_re_not_3)~=3))));
    
    % For 4
    idx_re_4 = find(test_r(:,12) == 4);
    idx_re_not_4 = find(test_r(:,12) ~= 4);
    idx_pr_4 = find(p_nb == 4);
    idx_pr_not_4 = find(p_nb ~= 4);
    L_4 = [length(find(p_nb(idx_re_4)==4)) length(find(p_nb(idx_re_not_4)~=4))...
        length(find(p_nb(idx_re_not_4)==4)) length(find(p_nb(idx_re_4)~=4))];
    L_4_rates = 100*[length(find(p_nb(idx_re_4)==4))/(length(find(p_nb(idx_re_4)==4))+length(find(p_nb(idx_re_4)~=4))) length(find(p_nb(idx_re_not_4)~=4))/(length(find(p_nb(idx_re_not_4)~=4))+length(find(p_nb(idx_re_not_4)==4)))...
        length(find(p_nb(idx_re_not_4)==4))/(length(find(p_nb(idx_re_not_4)==4))+length(find(p_nb(idx_re_not_4)~=4))) length(find(p_nb(idx_re_4)~=4))/(length(find(p_nb(idx_re_4)~=4))+length(find(p_nb(idx_re_4)==4)))];
    prec_4 = length(find(p_nb(idx_re_4)==4))/(length(find(p_nb(idx_re_4)==4))+length(find(p_nb(idx_re_not_4)==4)));
    
    f1_4 = (2*length(find(p_nb(idx_re_4)==4)))/(2*length(find(p_nb(idx_re_4)==4))+length(find(p_nb(idx_re_not_4)==4))+length(find(p_nb(idx_re_4)~=4)));
    
    Tot = length(find(p_nb(idx_re_4)==4)) + length(find(p_nb(idx_re_not_4)~=4)) + ...
        length(find(p_nb(idx_re_not_4)==4)) + length(find(p_nb(idx_re_4)~=4));
    
    acc_4 = (length(find(p_nb(idx_re_4)==4))+length(find(p_nb(idx_re_not_4)~=4)))/Tot;
    
    err_4 = (length(find(p_nb(idx_re_not_4)==4)) + length(find(p_nb(idx_re_4)~=4)))/Tot;
    
    tss_4 = (length(find(p_nb(idx_re_4)==4))/(length(find(p_nb(idx_re_4)==4))+length(find(p_nb(idx_re_4)~=4))))-...
        (length(find(p_nb(idx_re_not_4)==4))/(length(find(p_nb(idx_re_not_4)==4))+length(find(p_nb(idx_re_not_4)~=4))));
    
    % For 5
    idx_re_5 = find(test_r(:,12) == 5);
    idx_re_not_5 = find(test_r(:,12) ~= 5);
    idx_pr_5 = find(p_nb == 5);
    idx_pr_not_5 = find(p_nb ~= 5);
    L_5 = [length(find(p_nb(idx_re_5)==5)) length(find(p_nb(idx_re_not_5)~=5))...
        length(find(p_nb(idx_re_not_5)==5)) length(find(p_nb(idx_re_5)~=5))];
    L_5_rates = 100*[length(find(p_nb(idx_re_5)==5))/(length(find(p_nb(idx_re_5)==5))+length(find(p_nb(idx_re_5)~=5))) length(find(p_nb(idx_re_not_5)~=5))/(length(find(p_nb(idx_re_not_5)~=5))+length(find(p_nb(idx_re_not_5)==5)))...
        length(find(p_nb(idx_re_not_5)==5))/(length(find(p_nb(idx_re_not_5)==5))+length(find(p_nb(idx_re_not_5)~=5))) length(find(p_nb(idx_re_5)~=5))/(length(find(p_nb(idx_re_5)~=5))+length(find(p_nb(idx_re_5)==5)))];
    prec_5 = length(find(p_nb(idx_re_5)==5))/(length(find(p_nb(idx_re_5)==5))+length(find(p_nb(idx_re_not_5)==5)));
    
    f1_5 = (2*length(find(p_nb(idx_re_5)==5)))/(2*length(find(p_nb(idx_re_5)==5))+length(find(p_nb(idx_re_not_5)==5))+length(find(p_nb(idx_re_5)~=5)));
    
    Tot = length(find(p_nb(idx_re_5)==5)) + length(find(p_nb(idx_re_not_5)~=5)) + ...
        length(find(p_nb(idx_re_not_5)==5)) + length(find(p_nb(idx_re_5)~=5));
    
    acc_5 = (length(find(p_nb(idx_re_5)==5))+length(find(p_nb(idx_re_not_5)~=5)))/Tot;
    
    err_5 = (length(find(p_nb(idx_re_not_5)==5)) + length(find(p_nb(idx_re_5)~=5)))/Tot;
    
    tss_5 = (length(find(p_nb(idx_re_5)==5))/(length(find(p_nb(idx_re_5)==5))+length(find(p_nb(idx_re_5)~=5))))-...
        (length(find(p_nb(idx_re_not_5)==5))/(length(find(p_nb(idx_re_not_5)==5))+length(find(p_nb(idx_re_not_5)~=5))));
    
    % For 6
    idx_re_6 = find(test_r(:,12) == 6);
    idx_re_not_6 = find(test_r(:,12) ~= 6);
    idx_pr_6 = find(p_nb == 6);
    idx_pr_not_6 = find(p_nb ~= 6);
    L_6 = [length(find(p_nb(idx_re_6)==6)) length(find(p_nb(idx_re_not_6)~=6))...
        length(find(p_nb(idx_re_not_6)==6)) length(find(p_nb(idx_re_6)~=6))];
    L_6_rates = 100*[length(find(p_nb(idx_re_6)==6))/(length(find(p_nb(idx_re_6)==6))+length(find(p_nb(idx_re_6)~=6))) length(find(p_nb(idx_re_not_6)~=6))/(length(find(p_nb(idx_re_not_6)~=6))+length(find(p_nb(idx_re_not_6)==6)))...
        length(find(p_nb(idx_re_not_6)==6))/(length(find(p_nb(idx_re_not_6)==6))+length(find(p_nb(idx_re_not_6)~=6))) length(find(p_nb(idx_re_6)~=6))/(length(find(p_nb(idx_re_6)~=6))+length(find(p_nb(idx_re_6)==6)))];
    prec_6 = length(find(p_nb(idx_re_6)==6))/(length(find(p_nb(idx_re_6)==6))+length(find(p_nb(idx_re_not_6)==6)));
    
    f1_6 = (2*length(find(p_nb(idx_re_6)==6)))/(2*length(find(p_nb(idx_re_6)==6))+length(find(p_nb(idx_re_not_6)==6))+length(find(p_nb(idx_re_6)~=6)));
    
    Tot = length(find(p_nb(idx_re_6)==6)) + length(find(p_nb(idx_re_not_6)~=6)) + ...
        length(find(p_nb(idx_re_not_6)==6)) + length(find(p_nb(idx_re_6)~=6));
    
    acc_6 = (length(find(p_nb(idx_re_6)==6))+length(find(p_nb(idx_re_not_6)~=6)))/Tot;
    
    err_6 = (length(find(p_nb(idx_re_not_6)==6)) + length(find(p_nb(idx_re_6)~=6)))/Tot;
    
    tss_6 = (length(find(p_nb(idx_re_6)==6))/(length(find(p_nb(idx_re_6)==6))+length(find(p_nb(idx_re_6)~=6))))-...
        (length(find(p_nb(idx_re_not_6)==6))/(length(find(p_nb(idx_re_not_6)==6))+length(find(p_nb(idx_re_not_6)~=6))));
    
    % For 7
    idx_re_7 = find(test_r(:,12) == 7);
    idx_re_not_7 = find(test_r(:,12) ~= 7);
    idx_pr_7 = find(p_nb == 7);
    idx_pr_not_7 = find(p_nb ~= 7);
    L_7 = [length(find(p_nb(idx_re_7)==7)) length(find(p_nb(idx_re_not_7)~=7))...
        length(find(p_nb(idx_re_not_7)==7)) length(find(p_nb(idx_re_7)~=7))];
    L_7_rates = 100*[length(find(p_nb(idx_re_7)==7))/(length(find(p_nb(idx_re_7)==7))+length(find(p_nb(idx_re_7)~=7))) length(find(p_nb(idx_re_not_7)~=7))/(length(find(p_nb(idx_re_not_7)~=7))+length(find(p_nb(idx_re_not_7)==7)))...
        length(find(p_nb(idx_re_not_7)==7))/(length(find(p_nb(idx_re_not_7)==7))+length(find(p_nb(idx_re_not_7)~=7))) length(find(p_nb(idx_re_7)~=7))/(length(find(p_nb(idx_re_7)~=7))+length(find(p_nb(idx_re_7)==7)))];
    prec_7 = length(find(p_nb(idx_re_7)==7))/(length(find(p_nb(idx_re_7)==7))+length(find(p_nb(idx_re_not_7)==7)));
    
    f1_7 = (2*length(find(p_nb(idx_re_7)==7)))/(2*length(find(p_nb(idx_re_7)==7))+length(find(p_nb(idx_re_not_7)==7))+length(find(p_nb(idx_re_7)~=7)));
    
    Tot = length(find(p_nb(idx_re_7)==7)) + length(find(p_nb(idx_re_not_7)~=7)) + ...
        length(find(p_nb(idx_re_not_7)==7)) + length(find(p_nb(idx_re_7)~=7));
    
    acc_7 = (length(find(p_nb(idx_re_7)==7))+length(find(p_nb(idx_re_not_7)~=7)))/Tot;
    
    err_7 = (length(find(p_nb(idx_re_not_7)==7)) + length(find(p_nb(idx_re_7)~=7)))/Tot;
    
    tss_7 = (length(find(p_nb(idx_re_7)==7))/(length(find(p_nb(idx_re_7)==7))+length(find(p_nb(idx_re_7)~=7))))-...
        (length(find(p_nb(idx_re_not_7)==7))/(length(find(p_nb(idx_re_not_7)==7))+length(find(p_nb(idx_re_not_7)~=7))));
    
    % For 8
    idx_re_8 = find(test_r(:,12) == 8);
    idx_re_not_8 = find(test_r(:,12) ~= 8);
    idx_pr_8 = find(p_nb == 8);
    idx_pr_not_8 = find(p_nb ~= 8);
    L_8 = [length(find(p_nb(idx_re_8)==8)) length(find(p_nb(idx_re_not_8)~=8))...
        length(find(p_nb(idx_re_not_8)==8)) length(find(p_nb(idx_re_8)~=8))];
    L_8_rates = 100*[length(find(p_nb(idx_re_8)==8))/(length(find(p_nb(idx_re_8)==8))+length(find(p_nb(idx_re_8)~=8))) length(find(p_nb(idx_re_not_8)~=8))/(length(find(p_nb(idx_re_not_8)~=8))+length(find(p_nb(idx_re_not_8)==8)))...
        length(find(p_nb(idx_re_not_8)==8))/(length(find(p_nb(idx_re_not_8)==8))+length(find(p_nb(idx_re_not_8)~=8))) length(find(p_nb(idx_re_8)~=8))/(length(find(p_nb(idx_re_8)~=8))+length(find(p_nb(idx_re_8)==8)))];
    prec_8 = length(find(p_nb(idx_re_8)==8))/(length(find(p_nb(idx_re_8)==8))+length(find(p_nb(idx_re_not_8)==8)));
    
    f1_8 = (2*length(find(p_nb(idx_re_8)==8)))/(2*length(find(p_nb(idx_re_8)==8))+length(find(p_nb(idx_re_not_8)==8))+length(find(p_nb(idx_re_8)~=8)));
    
    Tot = length(find(p_nb(idx_re_8)==8)) + length(find(p_nb(idx_re_not_8)~=8)) + ...
        length(find(p_nb(idx_re_not_8)==8)) + length(find(p_nb(idx_re_8)~=8));
    
    acc_8 = (length(find(p_nb(idx_re_8)==8))+length(find(p_nb(idx_re_not_8)~=8)))/Tot;
    
    err_8 = (length(find(p_nb(idx_re_not_8)==8)) + length(find(p_nb(idx_re_8)~=8)))/Tot;
    
    tss_8 = (length(find(p_nb(idx_re_8)==8))/(length(find(p_nb(idx_re_8)==8))+length(find(p_nb(idx_re_8)~=8))))-...
        (length(find(p_nb(idx_re_not_8)==8))/(length(find(p_nb(idx_re_not_8)==8))+length(find(p_nb(idx_re_not_8)~=8))));
    
    disp('Naive Bayes');
    disp('-------------------------');
    disp('-------------------------');
    disp('TP TN FP FN');
    disp('-------------------------');
    L_all_nb{i+1} = [L_3;L_4;L_5;L_6;L_7;L_8];
    cell2mat(L_all_nb)
    disp('TPR TNR FPR FNR');
    disp('-------------------------');
    L_all_rates_nb{i+1} = [L_3_rates;L_4_rates;L_5_rates;L_6_rates;L_7_rates;L_8_rates];
    cell2mat(L_all_rates_nb)
    disp('Precision');
    disp('-------------------------');
    Prec_all_nb{i+1} = [prec_3;prec_4;prec_5;prec_6;prec_7;prec_8];
    cell2mat(Prec_all_nb)
    disp('F1-Score');
    disp('-------------------------');
    f1_nb{i+1} = [f1_3;f1_4;f1_5;f1_6;f1_7;f1_8];
    cell2mat(f1_nb)
    disp('Accuracy');
    disp('-------------------------');
    acc_nb{i+1} = [acc_3;acc_4;acc_5;acc_6;acc_7;acc_8];
    cell2mat(acc_nb)
    disp('Error-rate');
    disp('-------------------------');
    err_nb{i+1} = [err_3;err_4;err_5;err_6;err_7;err_8];
    cell2mat(err_nb)
    disp('TSS');
    disp('-------------------------');
    tss_nb{i+1} = [tss_3;tss_4;tss_5;tss_6;tss_7;tss_8];
    cell2mat(tss_nb)
end

%% 10-fold cross validation for red-wine Random Forest

k_size = round(size(r_newww,1)/10);

L_all_rf = cell(10,1);
L_all_rates_rf = cell(10,1);
Prec_all_rf = cell(10,1);
f1_rf = cell(10,1);
acc_rf = cell(10,1);
err_rf = cell(10,1);
tss_rf = cell(10,1);

for i = 0:9
   
    test_r = r_newww(i*k_size+1:(i+1)*k_size,:);
    if i == 0
        train_r = r_newww((i+1)*k_size+1:end,:);
    elseif i == 9
        train_r = r_newww(1:i*k_size,:);
    else
        train_r = [r_newww(1:i*k_size,:);r_newww((i+1)*k_size:end,:)];
    end
    
    RFModel = TreeBagger(50,train_r(:,1:11),train_r(:,12),'OOBPrediction','On',...
    'Method','classification');
    p_rf = predict(RFModel,test_r(:,1:11));
    p_rf = cell2mat(p_rf);
    p_rf = str2num(p_rf);

    % For 3
    idx_re_3 = find(test_r(:,12) == 3);
    idx_re_not_3 = find(test_r(:,12) ~= 3);
    idx_pr_3 = find(p_rf == 3);
    idx_pr_not_3 = find(p_rf ~= 3);
    L_3 = [length(find(p_rf(idx_re_3)==3)) length(find(p_rf(idx_re_not_3)~=3))...
        length(find(p_rf(idx_re_not_3)==3)) length(find(p_rf(idx_re_3)~=3))];
    L_3_rates = 100*[length(find(p_rf(idx_re_3)==3))/(length(find(p_rf(idx_re_3)==3))+length(find(p_rf(idx_re_3)~=3))) length(find(p_rf(idx_re_not_3)~=3))/(length(find(p_rf(idx_re_not_3)~=3))+length(find(p_rf(idx_re_not_3)==3)))...
        length(find(p_rf(idx_re_not_3)==3))/(length(find(p_rf(idx_re_not_3)==3))+length(find(p_rf(idx_re_not_3)~=3))) length(find(p_rf(idx_re_3)~=3))/(length(find(p_rf(idx_re_3)~=3))+length(find(p_rf(idx_re_3)==3)))];
    prec_3 = length(find(p_rf(idx_re_3)==3))/(length(find(p_rf(idx_re_3)==3))+length(find(p_rf(idx_re_not_3)==3)));
    
    f1_3 = (2*length(find(p_rf(idx_re_3)==3)))/(2*length(find(p_rf(idx_re_3)==3))+length(find(p_rf(idx_re_not_3)==3))+length(find(p_rf(idx_re_3)~=3)));
    
    Tot = length(find(p_rf(idx_re_3)==3)) + length(find(p_rf(idx_re_not_3)~=3)) + ...
        length(find(p_rf(idx_re_not_3)==3)) + length(find(p_rf(idx_re_3)~=3));
    
    acc_3 = (length(find(p_rf(idx_re_3)==3))+length(find(p_rf(idx_re_not_3)~=3)))/Tot;
    
    err_3 = (length(find(p_rf(idx_re_not_3)==3)) + length(find(p_rf(idx_re_3)~=3)))/Tot;
    
    tss_3 = (length(find(p_rf(idx_re_3)==3))/(length(find(p_rf(idx_re_3)==3))+length(find(p_rf(idx_re_3)~=3))))-...
        (length(find(p_rf(idx_re_not_3)==3))/(length(find(p_rf(idx_re_not_3)==3))+length(find(p_rf(idx_re_not_3)~=3))));
    
    % For 4
    idx_re_4 = find(test_r(:,12) == 4);
    idx_re_not_4 = find(test_r(:,12) ~= 4);
    idx_pr_4 = find(p_rf == 4);
    idx_pr_not_4 = find(p_rf ~= 4);
    L_4 = [length(find(p_rf(idx_re_4)==4)) length(find(p_rf(idx_re_not_4)~=4))...
        length(find(p_rf(idx_re_not_4)==4)) length(find(p_rf(idx_re_4)~=4))];
    L_4_rates = 100*[length(find(p_rf(idx_re_4)==4))/(length(find(p_rf(idx_re_4)==4))+length(find(p_rf(idx_re_4)~=4))) length(find(p_rf(idx_re_not_4)~=4))/(length(find(p_rf(idx_re_not_4)~=4))+length(find(p_rf(idx_re_not_4)==4)))...
        length(find(p_rf(idx_re_not_4)==4))/(length(find(p_rf(idx_re_not_4)==4))+length(find(p_rf(idx_re_not_4)~=4))) length(find(p_rf(idx_re_4)~=4))/(length(find(p_rf(idx_re_4)~=4))+length(find(p_rf(idx_re_4)==4)))];
    prec_4 = length(find(p_rf(idx_re_4)==4))/(length(find(p_rf(idx_re_4)==4))+length(find(p_rf(idx_re_not_4)==4)));
    
    f1_4 = (2*length(find(p_rf(idx_re_4)==4)))/(2*length(find(p_rf(idx_re_4)==4))+length(find(p_rf(idx_re_not_4)==4))+length(find(p_rf(idx_re_4)~=4)));
    
    Tot = length(find(p_rf(idx_re_4)==4)) + length(find(p_rf(idx_re_not_4)~=4)) + ...
        length(find(p_rf(idx_re_not_4)==4)) + length(find(p_rf(idx_re_4)~=4));
    
    acc_4 = (length(find(p_rf(idx_re_4)==4))+length(find(p_rf(idx_re_not_4)~=4)))/Tot;
    
    err_4 = (length(find(p_rf(idx_re_not_4)==4)) + length(find(p_rf(idx_re_4)~=4)))/Tot;
    
    tss_4 = (length(find(p_rf(idx_re_4)==4))/(length(find(p_rf(idx_re_4)==4))+length(find(p_rf(idx_re_4)~=4))))-...
        (length(find(p_rf(idx_re_not_4)==4))/(length(find(p_rf(idx_re_not_4)==4))+length(find(p_rf(idx_re_not_4)~=4))));
    
    % For 5
    idx_re_5 = find(test_r(:,12) == 5);
    idx_re_not_5 = find(test_r(:,12) ~= 5);
    idx_pr_5 = find(p_rf == 5);
    idx_pr_not_5 = find(p_rf ~= 5);
    L_5 = [length(find(p_rf(idx_re_5)==5)) length(find(p_rf(idx_re_not_5)~=5))...
        length(find(p_rf(idx_re_not_5)==5)) length(find(p_rf(idx_re_5)~=5))];
    L_5_rates = 100*[length(find(p_rf(idx_re_5)==5))/(length(find(p_rf(idx_re_5)==5))+length(find(p_rf(idx_re_5)~=5))) length(find(p_rf(idx_re_not_5)~=5))/(length(find(p_rf(idx_re_not_5)~=5))+length(find(p_rf(idx_re_not_5)==5)))...
        length(find(p_rf(idx_re_not_5)==5))/(length(find(p_rf(idx_re_not_5)==5))+length(find(p_rf(idx_re_not_5)~=5))) length(find(p_rf(idx_re_5)~=5))/(length(find(p_rf(idx_re_5)~=5))+length(find(p_rf(idx_re_5)==5)))];
    prec_5 = length(find(p_rf(idx_re_5)==5))/(length(find(p_rf(idx_re_5)==5))+length(find(p_rf(idx_re_not_5)==5)));
    
    f1_5 = (2*length(find(p_rf(idx_re_5)==5)))/(2*length(find(p_rf(idx_re_5)==5))+length(find(p_rf(idx_re_not_5)==5))+length(find(p_rf(idx_re_5)~=5)));
    
    Tot = length(find(p_rf(idx_re_5)==5)) + length(find(p_rf(idx_re_not_5)~=5)) + ...
        length(find(p_rf(idx_re_not_5)==5)) + length(find(p_rf(idx_re_5)~=5));
    
    acc_5 = (length(find(p_rf(idx_re_5)==5))+length(find(p_rf(idx_re_not_5)~=5)))/Tot;
    
    err_5 = (length(find(p_rf(idx_re_not_5)==5)) + length(find(p_rf(idx_re_5)~=5)))/Tot;
    
    tss_5 = (length(find(p_rf(idx_re_5)==5))/(length(find(p_rf(idx_re_5)==5))+length(find(p_rf(idx_re_5)~=5))))-...
        (length(find(p_rf(idx_re_not_5)==5))/(length(find(p_rf(idx_re_not_5)==5))+length(find(p_rf(idx_re_not_5)~=5))));
    
    % For 6
    idx_re_6 = find(test_r(:,12) == 6);
    idx_re_not_6 = find(test_r(:,12) ~= 6);
    idx_pr_6 = find(p_rf == 6);
    idx_pr_not_6 = find(p_rf ~= 6);
    L_6 = [length(find(p_rf(idx_re_6)==6)) length(find(p_rf(idx_re_not_6)~=6))...
        length(find(p_rf(idx_re_not_6)==6)) length(find(p_rf(idx_re_6)~=6))];
    L_6_rates = 100*[length(find(p_rf(idx_re_6)==6))/(length(find(p_rf(idx_re_6)==6))+length(find(p_rf(idx_re_6)~=6))) length(find(p_rf(idx_re_not_6)~=6))/(length(find(p_rf(idx_re_not_6)~=6))+length(find(p_rf(idx_re_not_6)==6)))...
        length(find(p_rf(idx_re_not_6)==6))/(length(find(p_rf(idx_re_not_6)==6))+length(find(p_rf(idx_re_not_6)~=6))) length(find(p_rf(idx_re_6)~=6))/(length(find(p_rf(idx_re_6)~=6))+length(find(p_rf(idx_re_6)==6)))];
    prec_6 = length(find(p_rf(idx_re_6)==6))/(length(find(p_rf(idx_re_6)==6))+length(find(p_rf(idx_re_not_6)==6)));
    
    f1_6 = (2*length(find(p_rf(idx_re_6)==6)))/(2*length(find(p_rf(idx_re_6)==6))+length(find(p_rf(idx_re_not_6)==6))+length(find(p_rf(idx_re_6)~=6)));
    
    Tot = length(find(p_rf(idx_re_6)==6)) + length(find(p_rf(idx_re_not_6)~=6)) + ...
        length(find(p_rf(idx_re_not_6)==6)) + length(find(p_rf(idx_re_6)~=6));
    
    acc_6 = (length(find(p_rf(idx_re_6)==6))+length(find(p_rf(idx_re_not_6)~=6)))/Tot;
    
    err_6 = (length(find(p_rf(idx_re_not_6)==6)) + length(find(p_rf(idx_re_6)~=6)))/Tot;
    
    tss_6 = (length(find(p_rf(idx_re_6)==6))/(length(find(p_rf(idx_re_6)==6))+length(find(p_rf(idx_re_6)~=6))))-...
        (length(find(p_rf(idx_re_not_6)==6))/(length(find(p_rf(idx_re_not_6)==6))+length(find(p_rf(idx_re_not_6)~=6))));
    
    % For 7
    idx_re_7 = find(test_r(:,12) == 7);
    idx_re_not_7 = find(test_r(:,12) ~= 7);
    idx_pr_7 = find(p_rf == 7);
    idx_pr_not_7 = find(p_rf ~= 7);
    L_7 = [length(find(p_rf(idx_re_7)==7)) length(find(p_rf(idx_re_not_7)~=7))...
        length(find(p_rf(idx_re_not_7)==7)) length(find(p_rf(idx_re_7)~=7))];
    L_7_rates = 100*[length(find(p_rf(idx_re_7)==7))/(length(find(p_rf(idx_re_7)==7))+length(find(p_rf(idx_re_7)~=7))) length(find(p_rf(idx_re_not_7)~=7))/(length(find(p_rf(idx_re_not_7)~=7))+length(find(p_rf(idx_re_not_7)==7)))...
        length(find(p_rf(idx_re_not_7)==7))/(length(find(p_rf(idx_re_not_7)==7))+length(find(p_rf(idx_re_not_7)~=7))) length(find(p_rf(idx_re_7)~=7))/(length(find(p_rf(idx_re_7)~=7))+length(find(p_rf(idx_re_7)==7)))];
    prec_7 = length(find(p_rf(idx_re_7)==7))/(length(find(p_rf(idx_re_7)==7))+length(find(p_rf(idx_re_not_7)==7)));
    
    f1_7 = (2*length(find(p_rf(idx_re_7)==7)))/(2*length(find(p_rf(idx_re_7)==7))+length(find(p_rf(idx_re_not_7)==7))+length(find(p_rf(idx_re_7)~=7)));
    
    Tot = length(find(p_rf(idx_re_7)==7)) + length(find(p_rf(idx_re_not_7)~=7)) + ...
        length(find(p_rf(idx_re_not_7)==7)) + length(find(p_rf(idx_re_7)~=7));
    
    acc_7 = (length(find(p_rf(idx_re_7)==7))+length(find(p_rf(idx_re_not_7)~=7)))/Tot;
    
    err_7 = (length(find(p_rf(idx_re_not_7)==7)) + length(find(p_rf(idx_re_7)~=7)))/Tot;
    
    tss_7 = (length(find(p_rf(idx_re_7)==7))/(length(find(p_rf(idx_re_7)==7))+length(find(p_rf(idx_re_7)~=7))))-...
        (length(find(p_rf(idx_re_not_7)==7))/(length(find(p_rf(idx_re_not_7)==7))+length(find(p_rf(idx_re_not_7)~=7))));
    
    % For 8
    idx_re_8 = find(test_r(:,12) == 8);
    idx_re_not_8 = find(test_r(:,12) ~= 8);
    idx_pr_8 = find(p_rf == 8);
    idx_pr_not_8 = find(p_rf ~= 8);
    L_8 = [length(find(p_rf(idx_re_8)==8)) length(find(p_rf(idx_re_not_8)~=8))...
        length(find(p_rf(idx_re_not_8)==8)) length(find(p_rf(idx_re_8)~=8))];
    L_8_rates = 100*[length(find(p_rf(idx_re_8)==8))/(length(find(p_rf(idx_re_8)==8))+length(find(p_rf(idx_re_8)~=8))) length(find(p_rf(idx_re_not_8)~=8))/(length(find(p_rf(idx_re_not_8)~=8))+length(find(p_rf(idx_re_not_8)==8)))...
        length(find(p_rf(idx_re_not_8)==8))/(length(find(p_rf(idx_re_not_8)==8))+length(find(p_rf(idx_re_not_8)~=8))) length(find(p_rf(idx_re_8)~=8))/(length(find(p_rf(idx_re_8)~=8))+length(find(p_rf(idx_re_8)==8)))];
    prec_8 = length(find(p_rf(idx_re_8)==8))/(length(find(p_rf(idx_re_8)==8))+length(find(p_rf(idx_re_not_8)==8)));
    
    f1_8 = (2*length(find(p_rf(idx_re_8)==8)))/(2*length(find(p_rf(idx_re_8)==8))+length(find(p_rf(idx_re_not_8)==8))+length(find(p_rf(idx_re_8)~=8)));
    
    Tot = length(find(p_rf(idx_re_8)==8)) + length(find(p_rf(idx_re_not_8)~=8)) + ...
        length(find(p_rf(idx_re_not_8)==8)) + length(find(p_rf(idx_re_8)~=8));
    
    acc_8 = (length(find(p_rf(idx_re_8)==8))+length(find(p_rf(idx_re_not_8)~=8)))/Tot;
    
    err_8 = (length(find(p_rf(idx_re_not_8)==8)) + length(find(p_rf(idx_re_8)~=8)))/Tot;
    
    tss_8 = (length(find(p_rf(idx_re_8)==8))/(length(find(p_rf(idx_re_8)==8))+length(find(p_rf(idx_re_8)~=8))))-...
        (length(find(p_rf(idx_re_not_8)==8))/(length(find(p_rf(idx_re_not_8)==8))+length(find(p_rf(idx_re_not_8)~=8))));
    
    disp('Random Forest');
    disp('-------------------------');
    disp('-------------------------');
    disp('TP TN FP FN');
    disp('-------------------------');
    L_all_rf{i+1} = [L_3;L_4;L_5;L_6;L_7;L_8];
    cell2mat(L_all_rf)
    disp('TPR TNR FPR FNR');
    disp('-------------------------');
    L_all_rates_rf{i+1} = [L_3_rates;L_4_rates;L_5_rates;L_6_rates;L_7_rates;L_8_rates];
    cell2mat(L_all_rates_rf)
    disp('Precision');
    disp('-------------------------');
    Prec_all_rf{i+1} = [prec_3;prec_4;prec_5;prec_6;prec_7;prec_8];
    cell2mat(Prec_all_rf)
    disp('F1-Score');
    disp('-------------------------');
    f1_rf{i+1} = [f1_3;f1_4;f1_5;f1_6;f1_7;f1_8];
    cell2mat(f1_rf)
    disp('Accuracy');
    disp('-------------------------');
    acc_rf{i+1} = [acc_3;acc_4;acc_5;acc_6;acc_7;acc_8];
    cell2mat(acc_rf)
    disp('Error-rate');
    disp('-------------------------');
    err_rf{i+1} = [err_3;err_4;err_5;err_6;err_7;err_8];
    cell2mat(err_rf)
    disp('TSS');
    disp('-------------------------');
    tss_rf{i+1} = [tss_3;tss_4;tss_5;tss_6;tss_7;tss_8];
    cell2mat(tss_rf)
end


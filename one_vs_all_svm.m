
function  er = one_vs_all_svm( train_x, train_y, test_x, test_y, c )

%    1对多的方法实现多分类svm
%    作者：王鹏举 
%    日期：2017年6月3日

[train_num, d] = size( train_x );
[test_num, n] = size( test_y );
weight = zeros( d, n );
bias = zeros( 1, n );

for i = 1:n
    train_y1 = ones(train_num, 1);
    [idx,~] = find(train_y(:, i) == 0);
    train_y1(idx) = -1;
    [weight(:,i), bias(n), ~] = trainSVM( train_x, train_y1, c );
end


%% 训练错误率
yy = test_x * weight + bias;
[~ , C1] = max(yy');
[~ , C2] = max(test_y');
err_num = sum(C1-C2 ~= 0);
er = err_num / test_num;


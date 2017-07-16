
function [ weight, bias, SV  ] = trainSVM( x, y, c )
%    二分类线性SVM算法
%    作者：王鹏举 
%    日期：2017年6月3日

H = (x*x').*sparse(y*y');
f = -ones(size(y'));
Aeq = y';
beq = 0;
lb = zeros(size(y'));
ub = c*ones(size(y'));
alpha = quadprog(H,f,[],[],Aeq,beq,lb,ub);
weight = x'*(alpha.*y);   %求出权值

alpha = roundn(alpha,-4);           %对alpha取4位有效数字 
SV = find((alpha~=0) & (alpha~=c)); %找到支持向量
bias = y(SV) - x(SV,:)*weight;
bias = mean(bias);  %由支持向量求出偏置，最后取平均值
end


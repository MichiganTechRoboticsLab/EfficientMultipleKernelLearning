clc
clear
clearvars -global alldata
global alldata

disp('Loading data...')
alldata = load('covtype.data');

alldata( alldata(:,end) == 3 ) = 1;
alldata( alldata(:,end) == 4 ) = 1;
alldata( alldata(:,end) == 6 ) = 1;
alldata( alldata(:,end) == 7 ) = 1;

alldata( alldata(:,end) == 2 ) = -1;
alldata( alldata(:,end) == 5 ) = -1;

% data = load('ionosphere');
% alldata =  data.X;
% y = double(cell2mat(data.Y));
% yvals = unique(y);
% y( y == yvals(1) ) = 1; y( y == yvals(2) ) = -1;
% alldata = [alldata y]; clear data yvals y;

disp('Starting GAMKL')

p = 1;
rbfmax = 3;
filename = 'first';
nsamps = 1; % one didn't work
nsamps = 0.5; % didn't work
nsamps = 0.25;

GAMKLpNystromFun_BIG(p, rbfmax, filename, nsamps)

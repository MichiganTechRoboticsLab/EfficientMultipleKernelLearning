% This code uses the GAMKLp approach as in the FUZZIEEE paper, but
% also implements the nystrom stuff
function GAMKLpNystromFun_BIG(p, rbfmax, filename, nsamps)
global alldata

% p - 1 or 2 (regularization)
% rbfmax - number >= 2 (max # of rbf kernels)
% alldata - input data (see commented out section below for how to form
%           alldata using 6 UCI datasets)
% filename - string containing the filename to save
% nsamps is nystrom sampling percentage

% for numks = 1:1
% for rbfk = 2:rbfmax
rbfk = rbfmax; % number of kernels
% polyks = [0 0 1 1];
% links = [0 1 0 1];
disp(['Nystrom sampling quantity: ' num2str(nsamps)]);
%%%%%%%%%%%SETUP
% p = 2; % Type of GAMKL (lp-norm)
% rbfk = 10; % number of RBF kernels
% polyk = polyks(numks); % number of polynomial kernels
polyk = 0;
% link = links(numks); % number of linear (Dot product) kernels
link = 0;
sigmoidk = 0; % number of sigmoidal kernels
m = rbfk + polyk + link + sigmoidk; % number of kernels
num_chrom = 31; % number of chromosomes (make it odd)
num_iter = 25; % # of epochs of GA
num_trials = 1%00; % # of runs
% num_trials = 10;
numCV = 5;
%%%%%%%%%
rbfsigs = 1/size(alldata,2);

[Ndata, num_features] = size(alldata);

Ntest = round(.2*Ndata);
Ntrain = Ndata - Ntest;

rbfsigs = [rbfsigs linspace(0.1 * rbfsigs, 10 * rbfsigs, m - 1)];

% Start trials here %%
for trial = 1:num_trials
disp(['Trial: ' num2str(trial)]);
TrialSTART = tic;
% Define training and testing data
pindex = randperm(Ndata);
% load('pind.mat')
X = alldata(pindex(1:Ntrain),1:end-1);
mx = mean(X); s = std(X)+1e-9;
X = bsxfun(@rdivide,bsxfun(@minus,X,mx),s);
y = alldata(pindex(1:Ntrain),end);

Xtest = alldata(pindex(Ntrain+1:end),1:end-1);
Xtest = bsxfun(@rdivide,bsxfun(@minus,Xtest,mx),s);
ytest = alldata(pindex(Ntrain+1:end),end);

% Initialize kernel weights
sigmaK = rand(m,num_chrom); % Randomly initialize m-1 kernel weights
% sigmaK(:,1) = seed;

denoms = sum(abs(sigmaK).^p).^(1/p);
for a = 1:num_chrom
    sigmaK(:,a) = sigmaK(:,a)./denoms(a);
end
clear denoms

% Initialize kernels
    
% Compute m kernels from data
%Kmat = zeros(size(X,1),size(X(si,:),1),m);

num_samples = round( nsamps * Ntrain / 100 )

si = randperm( Ntrain ); si=si(1:num_samples);
epsilon = eps( 'double' );

krqmem = Ntrain * num_samples * rbfk * 8 / 1e9;

fprintf('The 3d kernel matrix will require\n approximately %f GB\n', krqmem)

% prompt = 'Do you want to continue? y/n [y]: ';
% str = input(prompt,'s');
% if isempty(str)
%     str = 'y';
% end
% 
% if strcmp(str, 'n')
%     disp('aborted')
%     return
% end

% memory map here????
% Compute rbfk RBF kernels from training data
Kmat = zeros( 0, Ntrain, num_samples);
save('Kmat','Kmat','-v7.3');
clear Kmat

mf = matfile('Kmat.mat','Writable',true);
disp('Starting kernel calculations...')
for a = 1:rbfk
    mf.Kmat(a, :, :) = svkernel( 'rbf', X, X(si,:), rbfsigs(a) );
    disp(['Done with kernel ' num2str( a ) ' of ' num2str( rbfk )]);
end
       
%%
% sum together m kernels to get initial conglomerate kernel K and fitnesses
Cp = 1;
nk = rbfk;
fitnesses = zeros(num_chrom,1);
for a = 1:num_chrom
    disp(['Chromosome ' num2str( a ) ' of ' num2str( num_chrom ) ])
    Kt = sum(bsxfun(@times,mf.Kmat,reshape(sigmaK(:,a),1,1,nk)),3);
    disp('Eig decomp...')
    [V, D]=eig( Kt(si, :) );
    thres = size(Kt(:,:), 2) * norm(Kt(si, :)) * epsilon; tiL=max(find(diag(D)>thres));
    D = diag(1./sqrt(diag(D(1:tiL,1:tiL))));
    M = V(:,1:tiL) * D;
    Xlin = Kt * M;
    
    % get Nystrom svm results on individual kernels
%     acc = svmtrain(y,[(1:size(Kny,1))' Kny],['-q -t 4 -w1 ' num2str(Cp) ' -v ' num2str(numCV)]);
    disp('LibLinear pre-fitnesses')
    acc = train(y, sparse( Xlin ), ['-q -w1 ' num2str(Cp) ' -v ' num2str(numCV)]); %LIBLINEAR
    
    fitnesses(a) = acc(1);
    clear model acc Xlin M Kt
    disp('complete')
end

prefit = fitnesses;
prefit
% return
% plot(fitnesses)
%%
% Ready to start the GA
disp('Starting GA')
gaSTART = tic;
for iter = 1:num_iter
    
    % Choose survivors
    % Keep best individual
    [bestfit(trial,iter),bestind] = max(fitnesses);
    keep(:,1) = sigmaK(:,bestind);
    BESTFIT(trial) = bestfit(iter);
    
    % scale fitnesses into [0,1]
    fitnesses = fitnesses./sum(fitnesses);
    
    %Choose parents
    for a = 1:(num_chrom-1)/2
        %choose parent 1
        cumsum = 0;
        ind = 1;
        rtemp = rand;
        while(cumsum < rtemp)
            cumsum = cumsum + fitnesses(ind);
            ind = ind + 1;
        end
        parent1 = sigmaK(:,ind-1);
        %choose parent 2
        cumsum = 0;
        ind = 1;
        rtemp = rand;
        while(cumsum < rtemp)
            cumsum = cumsum + fitnesses(ind);
            ind = ind + 1;
        end
        parent2 = sigmaK(:,ind-1);
        
        % Crossover?
        pcross = 0.6;
        if(rand < pcross)
            %Crossover
            %choose crossover index
%             disp('crossover')
            crossInd = randi(m-1,1);
            keep(:,2*a) = [parent1(1:crossInd);parent2(crossInd+1:end)];
            keep(:,2*a+1) = [parent2(1:crossInd);parent1(crossInd+1:end)];
        else
            %No crossover
            keep(:,2*a) = parent1;
            keep(:,2*a+1) = parent2;
        end
    end
    sigmaK = keep;
    clear keep
    
    % Mutation
    pmutate = 0.05;
    for a = 2:num_chrom % don't mutate the best one
        % mutate chrom?
        if(rand<pmutate)
            for b = 1:m-1
                %mutate gene?
                if(rand<pmutate)
                    sigmaK(b,a) =  rand;
                end
            end
        end
    end
    %scale chromosomes to meet the p-norm requirement
    denoms = sum(abs(sigmaK).^p).^(1/p);
    for a = 1:num_chrom
        sigmaK(:,a) = sigmaK(:,a)./denoms(a);
    end
    clear denoms
    
    % Evaluate fitness
    for a = 1:num_chrom
        Kt = sum(bsxfun(@times,mf.Kmat,reshape(sigmaK(:,a),1,1,nk)),3);
        [V, D]=eig( Kt(si, :) );
        thres = size(Kt(:,:), 2) * norm(Kt(si, :)) * epsilon; tiL=max(find(diag(D)>thres));
        D = diag(1./sqrt(diag(D(1:tiL,1:tiL))));
        M = V(:,1:tiL) * D;
        Xlin = Kt * M;
               
        SVMTrainingSTART = tic;
%         acc = svmtrain(y,[(1:size(Kny,1))' Kny],['-q -t 4 -w1 ' num2str(Cp) ' -v ' num2str(numCV)]);
        disp('LibLinear in GA')
        acc = train(y, sparse( Xlin ), ['-q -w1 ' num2str(Cp) ' -v ' num2str(numCV)]); %LIBLINEAR

        % Time each svm training (to compare libsvm and liblinear)
        timer.svmtraining(a, iter, trial) = toc(SVMTrainingSTART);
%         [~,acc,~] = svmpredict(y,[(1:Ntrain)' K(:,:,a)],model);
        fitnesses(a) = acc(1);
        disp(['Chromosome ' num2str( a ) ' of ' num2str( num_chrom ) 'complete.'])
        clear model acc Kt Xlin M D V
    end
disp(['Iteration ' num2str( iter ) ' of ' num2str( num_iter ) 'complete.'])
end
timer.ga(trial) = toc(gaSTART); % times the length for the GA to complete
timer.training(trial) = toc(TrialSTART); % times the whole training time, including kernel building
% plot(bestfit(trial,:))
% figure,
% imagesc(sigmaK),colorbar
% ax = gca;
% ax.YTickLabel = {rbfsigs};
% ax.XTickLabel = {fitnesses};
% xlabel('Chromosome Fitness'),ylabel('RBF-\sigma'),title('Weights of kernels after GA')
%%
% Test the winner
TestSTART = tic;
[~,bestind] = max(fitnesses);
KtWIN = sum(bsxfun(@times,mf.Kmat,reshape(sigmaK(:,bestind),1,1,nk)),3);

[V, D]=eig( KtWIN(si, :) );
thres = size(KtWIN(:,:), 2) * norm(KtWIN(si, :)) * epsilon; tiL=max(find(diag(D)>thres));
D = diag(1./sqrt(diag(D(1:tiL,1:tiL))));
MWIN = V(:,1:tiL) * D;
XlinWIN = KtWIN * MWIN;

% model = svmtrain(y,[(1:size(Kny,1))' Kny],'-q -t 4');
disp('LibLinear training')
model = train(y, sparse( XlinWIN ), '-q');
% train_acc1CV = svmtrain(y,[(1:size(Kny,1))' Kny],['-q -t 4 -v ' num2str(numCV)]);
disp('LibLinear CV')
acc = train(y, sparse( XlinWIN ), ['-q -w1 ' num2str(Cp) ' -v ' num2str(numCV)]); %LIBLINEAR
% [trainlabels(:,trial),train_acc1,~] = svmpredict(y,[(1:size(Kny,1))' Kny],model);
disp('LibLinear predict training data')
[~, train_acc1,~] = svmpredict(y, sparse( XlinWIN ), model); %LIBLINEAR

% Test the winner on testing data
% Compute m kernels from training/testing data
for ii = 1 : rbfk,
    Ktest(:, :, ii) = svkernel( 'rbf', Xtest, X(si, :), rbfsigs(ii) );
    disp(['Done with kernel ' num2str( ii ) ' of ' num2str( rbfk )]);
end
        
KztestCombined = sum(bsxfun(@times,Ktest,reshape(sigmaK(:,bestind),1,1,nk)),3);

Xlintest = KztestCombined * MWIN;

predictSTART = tic;
% [testlabels(:,trial),test_acc1,~] = svmpredict(ytest,[(1:Ntest)' newK],model);
disp('LibLinear predict testing data')
[~,test_acc1,~] = predict(ytest, sparse( Xlintest ), model, '-q'); %LIBLINEAR
timer.predict(trial) = toc(predictSTART); % saves time for svmpredict to work

trial
test_acc(trial,nsamps) = test_acc1(1)/100;
train_acc(trial,nsamps) = train_acc1(1)/100;
bestsigmas(:,trial,nsamps) = sigmaK(:,bestind);
timer.testing(trial,nsamps) = toc(TestSTART); % saves the time it takes to form testing kernels and run svm predict

end
clear Kmat Kt M Xlin XlinWIN MWIN KtWIN Ktest KztestCombined

% train_acc1CV
train_acc'
test_acc'
% sigmaK(:,bestind)
% return
%%
disp('       Test          Train')
mean([test_acc;train_acc]')
std([test_acc;train_acc]')
save(['output/' filename],'test_acc', 'train_acc', 'bestsigmas', 'timer')
% disp([filename num2str(p) num2str(nsamps)])
pause(1)
clearvars -except datanum p numks nsamps rbfk
% end

end


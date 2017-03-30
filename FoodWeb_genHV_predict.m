% Dynamic Bayesian Network model with hidden variables
% depicting the food web dynamics of the Gotland Basin
% in the Baltic Sea
%
% Laura Uusitalo, 2017. laura.uusitalo@iki.fi

%%%%%%
% This model version
% - uses the first 35 years of data to learn the model, tries to predict 3
% years
% - implements a model that has a generic hidden variable (linked to all
% other variables in the model) 

N = 22; % number of variables in one time slice

% name the variables for easier access
HVGen = 1; 
FCod = 2; FSpr = 3; FHer = 4;  
RV = 5; Chla = 6; TSpring = 7; TSum = 8; 
SSBCod = 9; SSBSpr = 10; SSBHer = 11; 
Ps = 12; Tem = 13; Ac = 14;
Spr0y = 15; Her0y = 16; Cod0y = 17;
Spr1y=18; Her1y=19; 
Cod1y=20; Cod2y=21; Cod3y=22;

% DAG structure: 

% "intra" table encodes the structure of one time slice 
% See figure 5 in thesis for a graphical presentation
intra = zeros(N,N);% table to build in the dependencies
intra(FSpr, SSBSpr) = 1;
intra(FHer, SSBHer) = 1;
intra(FCod, SSBCod) = 1;
intra(RV, [Cod0y Ps]) = 1;
intra(Chla, [Ps Ac Tem]) = 1;
intra(TSpring, [Ps Ac Tem]) = 1;
intra(TSum, [Spr0y Her0y]) = 1;
intra(SSBSpr, [Spr0y Cod0y Ps Tem Ac]) = 1;
intra(SSBHer, [Her0y Ps Tem Ac]) = 1;
intra(SSBCod, [SSBSpr SSBHer Cod0y]) = 1;
intra(Ps, Cod0y) = 1;
intra(HVGen, (2:N)) = 1; % linked to all other variables

% "inter" encodes the dependencies between tme slices; 
% See figure 6 in the thesis for a graphical presentation
inter = zeros(N,N);
inter(HVGen,HVGen) = 1; %hidden variable linked to itself across time
inter(SSBHer, SSBHer) = 1;
inter(SSBSpr, SSBSpr) = 1;
inter(SSBCod, SSBCod) = 1;
inter(Her0y, Her1y) = 1;
inter(Her1y, SSBHer) = 1;
inter(Spr0y, Spr1y) = 1;
inter(Spr1y, SSBSpr) = 1;
inter(Cod0y, Cod1y) = 1;
inter(Cod1y, Cod2y) = 1;
inter(Cod2y, Cod3y) = 1;
inter(Cod3y, SSBCod) = 1;


% Read in the data 
% Missing values encoded as NaN, converted to empty cell
%the file needs to have the variables in the numbered order in columns,
%also HVs
data = csvread('data_genHV.csv', 1, 0, 'A2..V36'); % leave out colnames row, PLUS the 3 last years
data = num2cell(data);
[datlen datn] = size(data);
for i = 1:datlen
    for j = 1:datn
        if isnan(data{i,j})
            data{i,j}=[];
        end
    end
end

% Which nodes will be observed? 
onodes = [2:5,8:11,18:19,21]; 
dnodes = [ ]; %no discrete nodes
ns = ones(1,N);

% Define equivalence classes for the model variables:
%Equivalence classes are needed in order to learn the conditional
%probability tables from the data so that all data related to a variable,
%i.e. data from all years, is used to learn the distribution; the eclass
%specifies which variables "are the same".

% In the first year, all vars have their own eclasses;
% in the consecutive years, each variable belongs to the same eclass 
% with itself from the other time slices. 
% This is because due to the temporal dependencies, some of the variables have a
% different number of incoming arcs, and therefore cannot be in the same
% eclass. 

eclass1 = 1:N; % first time slice
eclass2 = (N+1):(2*N);% consecutive time slices
eclass = [eclass1 eclass2];
 
% Make the model
bnet = mk_dbn(intra, inter, ns, 'observed', onodes, 'discrete', dnodes, 'eclass1', eclass1, 'eclass2', eclass2);

% Loop over the EM-learning and pick the best model (based on
% log-likelihood) out of 1000 runs
% to avoid getting a result that got stuch to a poor local optimum
rng('shuffle') %init the random number generator based on time stamp
bestloglik = -inf; %initialize
for j = 1:100
    j
    % Set the priors: randomly drawn from N(0,1), with diagonal covariance matrices
    for i = 1:(2*N)
        bnet.CPD{i} = gaussian_CPD(bnet, i, 'cov_type', 'diag');
    end

    %Junction tree learning engine for parameter learning
    engine = jtree_unrolled_dbn_inf_engine(bnet, datlen);
    [bnet2, LLtrace] = learn_params_dbn_em(engine, {data'}, 'max_iter', 500); %tends to stop well before 50th iteration
    loglik = LLtrace(length(LLtrace));
    
    %when a better model is found, save
    if loglik > bestloglik
        bestloglik = loglik;
        bestbnet = bnet2;
            
    end
end

%save the bestbnet object
save('bestbnet_genHV_predict_diag.mat','bestbnet')


%
%PREDICTION
%

%%predict 3 years into the future, use the first 35 years as evidence
t=datlen+3 % 35+3

% SampleMarg function gives the mean and sd of each of the vars, each time step
margs=SampleMarg(bestbnet, data(1:datlen,:)',t);

%predict the fish variables that have been observed
SSBCodMu = []; 
SSBSprMu = []; 
SSBHerMu = []; 
Spr1yMu = []; 
Her1yMu = []; 
Cod2yMu = []; 

SSBCodSig = []; 
SSBSprSig = []; 
SSBHerSig = []; 
Spr1ySig = []; 
Her1ySig = []; 
Cod2ySig = []; 

%write the means and sds of the interest variables down for easier access
    for i = 36:t
        i
        %means
        SSBCodMu(i) = margs{9,i}.mu;
        SSBSprMu(i) = margs{10,i}.mu;
        SSBHerMu(i) = margs{11,i}.mu;
        Spr1yMu(i) = margs{18,i}.mu;
        Her1yMu(i) = margs{19,i}.mu;
        Cod2yMu(i) = margs{21,i}.mu;
        
        %sds
        SSBCodSig(i) = margs{9,i}.Sigma;
        SSBSprSig(i) = margs{10,i}.Sigma;
        SSBHerSig(i) = margs{11,i}.Sigma;
        Spr1ySig(i) = margs{18,i}.Sigma;
        Her1ySig(i) = margs{19,i}.Sigma;
        Cod2ySig(i) = margs{21,i}.Sigma;
        
    end
    
%save variables
save('SSBCodMu_genHV_Pred.txt','SSBCodMu','-ascii')
save('SSBCodSig_genHV_Pred.txt','SSBCodSig','-ascii')

save('SSBSprMu_genHV_Pred.txt','SSBSprMu','-ascii')
save('SSBSprSig_genHV_Pred.txt','SSBSprSig','-ascii')

save('SSBHerMu_genHV_Pred.txt','SSBHerMu','-ascii')
save('SSBHerSig_genHV_Pred.txt','SSBHerSig','-ascii')

save('Spr1yMu_genHV_Pred.txt','Spr1yMu','-ascii')
save('Spr1ySig_genHV_Pred.txt','Spr1ySig','-ascii')

save('Her1yMu_genHV_Pred.txt','Her1yMu','-ascii')
save('Her1ySig_genHV_Pred.txt','Her1ySig','-ascii')

save('Cod2yMu_genHV_Pred.txt','Cod2yMu','-ascii')
save('Cod2ySig_genHV_Pred.txt','Cod2ySig','-ascii')
% Dynamic Bayesian Network model with hidden variables
% depicting the food web dynamics of the Gotland Basin
% in the Baltic Sea
%
% Laura Uusitalo, 2017. laura.uusitalo@iki.fi

%%%%%%
% This model version
% - uses the first 35 years of data to fit the model, tries to predict 3
% years
% - implements a model that has a hidden cod variable (linked to cod
% variables) and a hidden clupeid variable (linked to all sprat & herring
% variables)

N = 23; % number of variables in one time slice

% name the variables for easier access
HVClu = 1; HVCod = 2; 
FCod = 3; FSpr = 4; FHer = 5;  
RV = 6; Chla = 7; TSpring = 8; TSum = 9; 
SSBCod = 10; SSBSpr = 11; SSBHer = 12; 
Ps = 13; Tem = 14; Ac = 15;
Spr0y = 16; Her0y = 17; Cod0y = 18;
Spr1y=19; Her1y=20; 
Cod1y=21; Cod2y=22; Cod3y=23;


% DAG structure: 

% "intra" table encodes the structure of one time slice 
% See figure 5 in manuscript for graphical presentation
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
intra(HVCod, [FCod Cod3y Cod2y SSBCod Cod0y Cod1y])=1; % Cod related only
intra(HVClu, [FSpr FHer SSBSpr SSBHer Spr0y Her0y Spr1y Her1y]) = 1; % All clupeid variables

% "inter" encodes the dependencies between tme slices; 
% See figure 6 in manuscript for graphical presentation
inter = zeros(N,N);
inter(HVClu, HVClu) = 1; % all hidden variables linked to themselves across time
inter(HVCod,HVCod) = 1; % all hidden variables linked to themselves across time
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


% Read in the data. 
% Missing vaues encoded as NaN, converted to empty cell
data = csvread('data_fishHV.csv', 1, 0, 'A2..W36'); % leave out colnames row AND last 3 years
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
onodes = [3:6,9:12,19:20,22]; 
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

%
% Loop over learning the parameters, pick the best model (based on
% log-likelihood):
%run the EM learning 100 times and pick the best loglikelihood model;
% this is because the algorithms seems to get stuck on local maximum 
rng('shuffle') %init the random number generator based on time stamp
bestloglik = -inf; %initialize
for j = 1:100
    j

    % Set the priors N(0,1), with diagonal covariance matrices.
    for i = 1:(2*N)
        bnet.CPD{i} = gaussian_CPD(bnet, i, 'cov_type', 'diag');
    end

    engine = jtree_unrolled_dbn_inf_engine(bnet, datlen);
    [bnet2, LLtrace] = learn_params_dbn_em(engine, {data'}, 'max_iter', 500);
    loglik = LLtrace(length(LLtrace));

    %when a better model is found, keep it
    if loglik > bestloglik
        bestloglik = loglik;
        bestbnet = bnet2;
    end
end

%save the bestbnet object
save('bestbnet_fishHVs_pred.mat','bestbnet')

%
%PREDICTION
%

%%predict 3 years into the future, use the first 35 years as evidence
t=datlen+3 % 35+3

% SampleMarg function gives the mean and sd of each of the vars, each time slice
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
        SSBCodMu(i) = margs{10,i}.mu;
        SSBSprMu(i) = margs{11,i}.mu;
        SSBHerMu(i) = margs{12,i}.mu;
        Spr1yMu(i) = margs{19,i}.mu;
        Her1yMu(i) = margs{20,i}.mu;
        Cod2yMu(i) = margs{22,i}.mu;
        
        %sigmas
        SSBCodSig(i) = margs{10,i}.Sigma;
        SSBSprSig(i) = margs{11,i}.Sigma;
        SSBHerSig(i) = margs{12,i}.Sigma;
        Spr1ySig(i) = margs{19,i}.Sigma;
        Her1ySig(i) = margs{20,i}.Sigma;
        Cod2ySig(i) = margs{22,i}.Sigma;
        
    end
    
%save variables
save('SSBCodMu_fishHVs_Pred.txt','SSBCodMu','-ascii')
save('SSBCodSig_fishHVs_Pred.txt','SSBCodSig','-ascii')

save('SSBSprMu_fishHVs_Pred.txt','SSBSprMu','-ascii')
save('SSBSprSig_fishHVs_Pred.txt','SSBSprSig','-ascii')

save('SSBHerMu_fishHVs_Pred.txt','SSBHerMu','-ascii')
save('SSBHerSig_fishHVs_Pred.txt','SSBHerSig','-ascii')

save('Spr1yMu_fishHVs_Pred.txt','Spr1yMu','-ascii')
save('Spr1ySig_fishHVs_Pred.txt','Spr1ySig','-ascii')

save('Her1yMu_fishHVs_Pred.txt','Her1yMu','-ascii')
save('Her1ySig_fishHVs_Pred.txt','Her1ySig','-ascii')

save('Cod2yMu_fishHVs_Pred.txt','Cod2yMu','-ascii')
save('Cod2ySig_fishHVs_Pred.txt','Cod2ySig','-ascii')
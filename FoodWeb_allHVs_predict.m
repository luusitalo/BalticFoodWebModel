% Dynamic Bayesian Network model with hidden variables
% depicting the food web dynamics of the Gotland Basin
% in the Baltic Sea
%
% Laura Uusitalo, 2017. laura.uusitalo@iki.fi

%%%%%%
% This model version
% - leaves 3 last years of data out, to be the test data 
% - implements a model that has both a generic hidden variable (linked to all
% other variables in the model) and a hidden cod variable (linked to cod
% variables) as well as a hidden clupeid variable (linked to all sprat & herring
% variables)

N = 24; % number of variables in one time slice

% name the variables for easier access
HVGen = 1; HVClu = 2; HVCod = 3; 
FCod = 4; FSpr = 5; FHer = 6;  
RV = 7; Chla = 8; TSpring = 9; TSum = 10; 
SSBCod = 11; SSBSpr = 12; SSBHer = 13; 
Ps = 14; Tem = 15; Ac = 16;
Spr0y = 17; Her0y = 18; Cod0y = 19;
Spr1y=20; Her1y=21; 
Cod1y=22; Cod2y=23; Cod3y=24;


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
intra(HVGen, (4:N)) = 1; % linked to all, EXCEPT the other "generic" HVs
intra(HVCod, [FCod SSBCod Cod0y Cod1y Cod2y Cod3y]) = 1; % Cod related only
intra(HVClu, [FSpr FHer SSBSpr SSBHer Spr0y Her0y Spr1y Her1y]) = 1; % All clupeid variables

% "inter" encodes the dependencies between tme slices; 
% See figure 6 in manuscript for graphical presentation
inter = zeros(N,N);
inter(HVGen,HVGen) = 1; %all hidden variables linked to themselves actoss time
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
data = csvread('data.csv', 1, 0, 'A2..X36'); % leave out colnames row AND 3 last years
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
onodes = [4:7,10:13,20:21,23]; 
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


% Loop over EM learning 100 times, keep the best model (based on
% log-likelihood, to avoid getting a result that's stuck to a poor local
% optimum
rng('shuffle') %init the random number generator based on time stamp
bestloglik = -inf; %initialize
for j = 1:100
    j

    % Set the priors random from N(0,1), with diagonal covariance matrices.
    for i = 1:(2*N)
        bnet.CPD{i} = gaussian_CPD(bnet, i, 'cov_type', 'diag');
    end
    
    %Junction tree learning engine for parameter learning
    engine = jtree_unrolled_dbn_inf_engine(bnet, datlen);
    [bnet2, LLtrace] = learn_params_dbn_em(engine, {data'}, 'max_iter', 500);   
    loglik = LLtrace(length(LLtrace));
    
    % when a better model is found, keep it 
    if loglik > bestloglik
        bestloglik = loglik;
        bestbnet = bnet2;
    end
end

%save the bestbnet object
save('bestbnet_allHVs_pred.mat','bestbnet')

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
        SSBCodMu(i) = margs{11,i}.mu;
        SSBSprMu(i) = margs{12,i}.mu;
        SSBHerMu(i) = margs{13,i}.mu;
        Spr1yMu(i) = margs{20,i}.mu;
        Her1yMu(i) = margs{21,i}.mu;
        Cod2yMu(i) = margs{23,i}.mu;
        
        %sds
        SSBCodSig(i) = margs{11,i}.Sigma;
        SSBSprSig(i) = margs{12,i}.Sigma;
        SSBHerSig(i) = margs{13,i}.Sigma;
        Spr1ySig(i) = margs{20,i}.Sigma;
        Her1ySig(i) = margs{21,i}.Sigma;
        Cod2ySig(i) = margs{23,i}.Sigma;
        
    end
    
%save variables
save('SSBCodMu_allHVs_Pred.txt','SSBCodMu','-ascii')
save('SSBCodSig_allHVs_Pred.txt','SSBCodSig','-ascii')

save('SSBSprMu_allHVs_Pred.txt','SSBSprMu','-ascii')
save('SSBSprSig_allHVs_Pred.txt','SSBSprSig','-ascii')

save('SSBHerMu_allHVs_Pred.txt','SSBHerMu','-ascii')
save('SSBHerSig_allHVs_Pred.txt','SSBHerSig','-ascii')

save('Spr1yMu_allHVs_Pred.txt','Spr1yMu','-ascii')
save('Spr1ySig_allHVs_Pred.txt','Spr1ySig','-ascii')

save('Her1yMu_allHVs_Pred.txt','Her1yMu','-ascii')
save('Her1ySig_allHVs_Pred.txt','Her1ySig','-ascii')

save('Cod2yMu_allHVs_Pred.txt','Cod2yMu','-ascii')
save('Cod2ySig_allHVs_Pred','Cod2ySig','-ascii')
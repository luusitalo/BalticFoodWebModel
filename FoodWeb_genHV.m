% Dynamic Bayesian Network model with hidden variables
% depicting the food web dynamics of the Gotland Basin
% in the Baltic Sea
%
% Laura Uusitalo, 2017. laura.uusitalo@iki.fi

%%%%%%
% This model version
% - uses all the data to fit the model
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
%intra(HVGen, (2:N)) = 1; % linked to all, ALSO the other "generic" HVs
intra(HVGen, (2:N)) = 1; % linked to all, EXCEPT the other "generic" HVs

% "inter" encodes the dependencies between tme slices; 
% See figure 6 in manuscript for graphical presentation
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
% Missing vaues encoded as NaN, converted to empty cell
data = csvread('data_genHV.csv', 1, 0); % leave out colnames row
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

%
% Loop over the EM learning 100 times, keep the best model (based on
% log-likelihood), to avoid getting a model that has got stuck to a por local optimum
rng('shuffle') %init the random number generator based on time stamp
bestloglik = -inf; %initialize
for j = 1:100
    j

    % Set the priors N(0,1), with diagonal covariance matrices.
    for i = 1:(2*N)
        bnet.CPD{i} = gaussian_CPD(bnet, i, 'cov_type', 'diag');
    end

    %Junction tree learning engine for parameter learning
    engine = jtree_unrolled_dbn_inf_engine(bnet, datlen);
    [bnet2, LLtrace] = learn_params_dbn_em(engine, {data'}, 'max_iter', 500);
    loglik = LLtrace(length(LLtrace));
    
    %when a better model is found, store it
    if loglik > bestloglik
        bestloglik = loglik;
        bestbnet = bnet2;
            
    end
end

%save the bestbnet object
save('/home/laurau/matlab/own/bestbnet_genHV_diag.mat','bestbnet')


t=datlen % 38 years

% mean and sd of each of the vars, each time slice
margs=SampleMarg(bestbnet,data(1:datlen,:)',t);

GenHVMu=[];% Generic HV
GenHVSig=[];% Generic HV
     
%write the means and sds of the interest variables down for easier access
for i = 1:t
    i
    %means
    GenHVMu(i) = margs{1,i}.mu;
    
    %sigma
    GenHVSig(i) = margs{1,i}.Sigma;
    end

    %save variables for plotting in R
    save('GenHVMu_onlyGenHV.txt','GenHVMu','-ascii')
    save('GenHVSig_onlyGenHV.txt','GenHVSig','-ascii')
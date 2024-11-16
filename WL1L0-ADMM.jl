# WL1L0-ADMM using ProximalOperators with constant learning rate
# Bayesian optimization for tuning alpha and lambda
    
# Load the required packages (if they are not installed, please install them before loading)
using BayesianOptimization, GaussianProcesses,Statistics
using DelimitedFiles,LinearAlgebra
using Random
using ProximalOperators

# Read QTLMAS2010 data
X = readdlm("QTLMAS2010ny012.csv",',');

ytot = (X[:,1].-mean(X[:,1])) # Mean-center the response
ytrain = ytot[1:2326]
Xtest= X[2327:size(X)[1],2:size(X)[2]]
ytest = ytot[2327:size(X)[1]]
Xtrain = X[1:2326,2:size(X)[2]]
p = size(Xtrain)[2] # number of variables

# One-hot encode the training data
Xtrain0 = copy(Xtrain)
Xtrain1 = copy(Xtrain)
Xtrain2 = copy(Xtrain)
Xtrain0[Xtrain0.==1] .= 2
Xtrain0[Xtrain0.==0] .= 1
Xtrain0[Xtrain0.==2] .= 0
Xtrain1[Xtrain1.==2] .= 0
Xtrain2[Xtrain2.==1] .= 0
Xtrain2[Xtrain2.==2] .= 1
Xtrain = hcat(Xtrain0,Xtrain1,Xtrain2)
# Set unimportant allocations to zero
Xtrain0 = 0
Xtrain1 = 0
Xtrain2 = 0
X = 0
p = size(Xtrain)[2] # Number of variables after concatenation

# One hot encode the test data
Xtest0 = copy(Xtest)
Xtest1 = copy(Xtest)
Xtest2 = copy(Xtest)
Xtest0[Xtest0.==1] .= 2
Xtest0[Xtest0.==0] .= 1
Xtest0[Xtest0.==2] .= 0
Xtest1[Xtest1.==2] .= 0
Xtest2[Xtest2.==1] .= 0
Xtest2[Xtest2.==2] .= 1
Xtest = hcat(Xtest0,Xtest1,Xtest2)
# Set unimportant allocations to zero
Xtest0 = 0
Xtest1 = 0
Xtest2 = 0


# Calculate covariances to initialize u and v
covar = cov(Xtrain,ytrain)

# Convergence tolerance of WL1L0-ADMM
tol=5e-4
# Maximum number of WL1L0-ADMM iterations
maxit=5000

# Singular value decomposition for learning rate gamma and delta (here, gamma = delta)
U, S, V = svd(Xtrain) 
σmax = S[1] 
gam = 1 / σmax

# The WL1L0-ADMM function for BO
WL1L0_ADMM_bo(par::Vector) = WL1L0_ADMM_bo(par[1],par[2]);
function WL1L0_ADMM_bo(par1,par2)
  u = zero(Xtrain[1,:])
  u = covar[:,1]*0.0001
  v = zero(Xtrain[1,:])
  v = covar[:,1]*0.0001
  uvcurr = zero(Xtrain[1,:])
  c = zero(Xtrain[1,:])
  d = zero(Xtrain[1,:])
  m = zero(c)
  w = zero(d)
  alp = par1
  lam = par2
  lamalp1 = lam*alp
  lamalp2 = lam*(1.0-alp)
  hL = LeastSquares(Xtrain, ytrain) # Loss function L1
  fL = Translate(hL, v) # Translation function L1
  gL = NormL1(lamalp1) # Regularization function L1
  hR = LeastSquares(Xtrain, ytrain) # Loss function L0
  fR = Translate(hR, u) # Translation function L0
  gR = NormL0(lamalp2) # Regularization function L0
  for it = 1:maxit
    uvcurr = u + v
    # ADMM perform f-update step L1
    prox!(c, fL, u - m, gam) 
    # ADMM perform g-update step L1
    prox!(u, gL, c + m, gam)    
    # Dual update L1
    m .+= c - u
    # ADMM perform f-update step L0
    prox!(d, fR, v - w, gam) 
    # ADMM perform g-update step L0
    prox!(v, gR, d + w, gam) 
    # Stopping criterion for ADMMM
    dualres = (u + v) - uvcurr
    reldualres = dualres/(norm(((u + v) + uvcurr)/2))
    if it % 5 == 2 && (norm(reldualres) <= tol)
      break
    end
    # Dual update L0
    w .+= d - v
  end
  Ytestpred = Xtest*(u+v) # Test predictions
  MSEtest = (norm(Ytestpred.-ytest)^2)/length(ytest) # Test MSE
  return MSEtest    
end

# Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 20 steps
modeloptimizer = MAPGPOptimizer(every = 20, noisebounds = [-1.,10.],
    kernbounds = [[-3., -3., 0.], [6., 8., 8.]],
    maxeval = 40)

model = ElasticGPE(2, mean = MeanConst(0.), kernel = SEArd([2.,3.], 1.),
  logNoise = 4., capacity = 1000)

# optimize alpha and lambda using BO
opt_WL1L0_ADMM = BOpt(par -> WL1L0_ADMM_bo(par[1], par[2]), model, 
  MutualInformation(), modeloptimizer,
  [0.01,0.001], [0.99,500.0], repetitions = 4, maxiterations = 250,
  sense = Min,
  verbosity = Progress)
  res_WL1L0_ADMM = boptimize!(opt_WL1L0_ADMM)

# Now use the optimed alp=par1 and lamb=par2 in the WL1L0_ADMM_bo function
function WL1L0_ADMM_bo(par1,par2)
  u = zero(Xtrain[1,:])
  u = covar[:,1]*0.0001
  v = zero(Xtrain[1,:])
  v = covar[:,1]*0.0001
  uvcurr = zero(Xtrain[1,:]) 
  c = zero(Xtrain[1,:])
  d = zero(Xtrain[1,:])
  m = zero(c)
  w = zero(d)
  alp = par1 
  lam = par2
  lamalp1 = lam*alp
  lamalp2 = lam*(1.0-alp)
  hL = LeastSquares(Xtrain, ytrain) 
  fL = Translate(hL, v)
  gL = NormL1(lamalp1)
  hR = LeastSquares(Xtrain, ytrain)
  fR = Translate(hR, u)
  gR = NormL0(lamalp2)
  for it = 1:maxit
    uvcurr = u + v
    prox!(c, fL, u - m, gam) 
    prox!(u, gL, c + m, gam)    
    m .+= c - u
    prox!(d, fR, v - w, gam)  
    prox!(v, gR, d + w, gam) 
    dualres = (u + v) - uvcurr
    reldualres = dualres/(norm(((u + v) + uvcurr)/2))
    if it % 5 == 2 && (norm(reldualres) <= tol)
      break
    end
    w .+= d - v
  end
  Ytestpred = Xtest*(u+v) 
  MSEtest = (norm(Ytestpred.-ytest)^2)/length(ytest) 
  result_list = [MSEtest, u + v] # List of test MSE and regression coefficients (u+v)
  return result_list
end

# Run WL1L0-ADMM with timing
@time result_WL1L0_ADMM = WL1L0_ADMM_bo(res_WL1L0_ADMM[2][1], res_WL1L0_ADMM[2][2])

# Count number of non-zeros regression coefficients
nonzeros_WL1L0_ADMM  = count(x -> x != 0,result_WL1L0_ADMM[2])
nonzeros_WL1L0_ADMM
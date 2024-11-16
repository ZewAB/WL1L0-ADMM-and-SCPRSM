# WL1L0-SCPRSM using ProximalOperators with constant learning rate
# Bayesian optimization for tuning alpha, lambda, and the relaxation factor
    
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

# Convergence tolerance of WL1L0-SCPRSM
tol=5e-4
# Maximum number of WL1L0-SCPRSM iterations
maxit=5000

# Singular value decomposition for learning rate gamma and delta (here, gamma = delta)
U, S, V = svd(Xtrain) 
σmax = S[1] 
gam = 1 / σmax

# The WL1L0-SCPRSM function for BO
WL1L0_SCPRSM_bo(par::Vector) = WL1L0_SCPRSM_bo(par[1],par[2],par[3]);
function WL1L0_SCPRSM_bo(par1,par2,par3)
  u = zero(Xtrain[1,:])
  u = covar[:,1]*0.0001
  v = zero(Xtrain[1,:])
  v = covar[:,1]*0.0001
  uvcurr = zero(Xtrain[1,:]) 
  c = zero(Xtrain[1,:])
  d = zero(Xtrain[1,:])
  m = zero(c)
  m2 = zero(c)
  w = zero(d)
  alp = par1
  lam = par2
  r = par3 # Relaxation factor
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
    # SCPRSM perform f-update step L1
    prox!(c, fL, u - m, gam) 
    # First dual update L1
    m .+= r*(c - u)
    # SCPRSM perform g-update step L1
    prox!(u, gL, c + m, gam)    
    # Second dual update L1
    m2 .+= r*(c - u)
    # SCPRSM perform f-update step L0
    prox!(d, fR, v - w, gam) 
    # First dual update L0
     w .+= r*(d - v)
    # SCPRSM perform g-update step L0
    prox!(v, gR, d + w, gam) 
    # Stopping criterion for SCPRSM
    dualres = (u + v) - uvcurr
    reldualres = dualres/(norm(((u + v) + uvcurr)/2))
    if it % 5 == 2 && (norm(reldualres) <= tol)
      break
    end
    # Second dual update L0
    w2 .+= r*(d - v)
  end
  Ytestpred = Xtest*(u+v) # Test predictions
  MSEtest = (norm(Ytestpred.-ytest)^2)/length(ytest) # Test MSE
  return MSEtest    
end

# Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 20 steps
modeloptimizer = MAPGPOptimizer(every = 30, noisebounds = [-1.,10.],
    kernbounds = [[-3., -3., -3, 0.], [6., 8., 8., 8.]],
    maxeval = 40)
model = ElasticGPE(3, mean = MeanConst(0.), kernel = SEArd([2.,3.,3.], 1.),
  logNoise = 4., capacity = 500)

# optimize alpha,lambda, and r using BO
opt_WL1L0_SCPRSM = BOpt(par -> WL1L0_SCPRSM_bo(par[1], par[2], par[3]), model, 
  MutualInformation(), modeloptimizer,
  [0.0001,0.0001,0.0001], [0.999,500.0,1.0], repetitions = 4, maxiterations = 250,
  sense = Min,
  verbosity = Progress)
  res_WL1L0_SCPRSM = boptimize!(opt_WL1L0_SCPRSM)

# Now use the optimed alp=lam1 and lamb=lam2 in the WL1L0_SCPRSM_bo function
function WL1L0_SCPRSM_bo(par1,par2,par3)
  u = zero(Xtrain[1,:])
  u = covar[:,1]*0.0001
  v = zero(Xtrain[1,:])
  v = covar[:,1]*0.0001
  uvcurr = zero(Xtrain[1,:]) 
  c = zero(Xtrain[1,:])
  d = zero(Xtrain[1,:])
  m = zero(c)
  m2 = zero(c)
  w = zero(d)
  w2 = zero(d)
  alp = par1
  lam = par2
  r = par3 
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
    m .+= r*(c - u)
    prox!(u, gL, c + m, gam)    
    m2 .+= r*(c - u)
    prox!(d, fR, v - w, gam) 
     w .+= r*(d - v)
    prox!(v, gR, d + w, gam) 
    dualres = (u + v) - uvcurr
    reldualres = dualres/(norm(((u + v) + uvcurr)/2))
    if it % 5 == 2 && (norm(reldualres) <= tol)
      break
    end
    w2 .+= r*(d - v)
  end
  Ytestpred = Xtest*(u+v)
  MSEtest = (norm(Ytestpred.-ytest)^2)/length(ytest) 
  result_list = [MSEtest, u + v] # List of test MSE and regression coefficients
  return result_list
end

# Run WL1L0-SCPRSM with timing
@time result_WL1L0_SCPRSM = WL1L0_SCPRSM_bo(res_WL1L0_SCPRSM[2][1], res_WL1L0_SCPRSM[2][2],res_WL1L0_SCPRSM[2][3])

# Count number of non-zeros regression coefficients
nonzeros_WL1L0_SCPRSM  = count(x -> x != 0,result_WL1L0_SCPRSM[2])
nonzeros_WL1L0_SCPRSM
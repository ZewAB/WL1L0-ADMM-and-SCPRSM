# WL1L0-ADMM and WL1L0-SCPRSM

WL1L0-ADMM and WL1L0-SCPRSM both contain Julia code for the proximal optimization of the weighted $L^1$ and $L^0$ regularizers, as presented in Berkessa and Waldmann (2024) and published in *Transactions on Machine Learning Research (TMLR)*. 

- **WL1L0-ADMM** implements the proximal ADMM method.
- **WL1L0-SCPRSM** implements the proximal SCPRSM method.

## Data Requirements

The code reads the data file **QTLMAS2010ny012.csv** from the local working directory. This data is available in the [AUTALASSO directory](https://github.com/patwa67/AUTALASSO/blob/master/QTLMAS2010ny012.zip). 

You need to:
1. Download the file.
2. Extract and save it to your local working directory.

### Data Format
- **y-variable (phenotype)**: Located in the first column.
- **x-variables (SNPs; coded as 0,1,2)**: Located in the following columns (comma-separated).
- **Data partitioning**: 
  - Training data: Generations 1-4.
  - Test data: Generation 5.

---

## Outputs for WL1L0-ADMM

1. **`res_WL1L0_ADMM = boptimize!(opt_WL1L0_ADMM)`**
   - Produces the minimum test MSE with regularization parameters `alpha` and `lambda` for the test MSE minimizer.

2. **`@time result_WL1L0_ADMM = WL1L0_ADMM_bo(res_WL1L0_ADMM[2][1], res_WL1L0_ADMM[2][2])`**
   - Produces timing and a list containing:
     - Minimum test MSE.
     - Sum of the regression coefficients.

3. **`nonzeros_WL1L0_ADMM`**
   - Outputs the number of non-zero coefficients.

---

## Outputs for WL1L0-SCPRSM

1. **`res_WL1L0_SCPRSM = boptimize!(opt_WL1L0_SCPRSM)`**
   - Produces the minimum test MSE with regularization parameters `alpha`, `lambda`, and the relaxation factor for the test MSE minimizer.

2. **`@time result_WL1L0_SCPRSM = WL1L0_SCPRSM_bo(res_WL1L0_SCPRSM[2][1], res_WL1L0_SCPRSM[2][2], res_WL1L0_SCPRSM[2][3])`**
   - Produces timing and a list containing:
     - Minimum test MSE.
     - Sum of the regression coefficients.

3. **`nonzeros_WL1L0_SCPRSM = count(x -> x != 0, result_WL1L0_SCPRSM[2])`**
   - Outputs the number of non-zero coefficients.

---

## Notes

The rest of the code usage for the pig dataset and mice dataset is straightforward and as described in the accompanying paper. Use the same code for **WL1L0-ADMM** and **WL1L0-SCPRSM** with the corresponding data (e.g., pig dataset or mice dataset). 

### Cross-Validation
1. Perform cross-validation five times using the `randperm()` function to generate training and test sets.
2. For each fold, compute:
   - Regularization parameters.
   - Computational runtime.
   - Evaluation metrics (MSE).
   - Number of non-zeros.

3. Average all parameters across the five folds to provide a comprehensive summary of performance.

---

## References

- Berkessa and Waldmann (2024), *Weighted $L^1$ and $L^0$ Regularization Using Proximal Operator Splitting Methods*, *Transactions on Machine Learning Research (TMLR)*.

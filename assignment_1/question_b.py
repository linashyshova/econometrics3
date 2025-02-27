'''
Filename: question_b.py

Purpose: Carry out the within estimation and report the estimates, including standard errors and t-statistics,
but also the coefficient of determination R2and your estimate of sigma_heta^2
    
Date:
    27 February 2025  
'''

###########################################################
### Imports
import numpy as np
import pandas as pd

# global variables
y_var = "lwage"
x_vars = ["exp", "wks", "bluecol", "ind", "south", "smsa", "married", "union"]

###########################################################


def within_regression(y, X , N=595, T=7):
    I_T = np.ones((T, 1)) 
    I_N = np.eye(N) 
    I_NT = np.eye(N*T)
    D = np.kron(I_N, I_T)  
    M_D = I_NT - D @ np.linalg.inv(D.T @ D) @ D.T
    
    # Differenced y and X per individual
    y_tilde = M_D @ y
    X_tilde = M_D @ X
    
    XtX_inv = np.linalg.inv(X_tilde.T @ X_tilde)
    beta_W = XtX_inv @ (X_tilde.T @ y_tilde)
    
    # Compute residuals
    u_i = y_tilde - X_tilde @ beta_W
    sigma2_heta = (1 / (N*(T-1)  - X.shape[1])) * np.sum(u_i**2)
    
    # Compute standard errors
    s2_W = sigma2_heta
    Sigma_W = XtX_inv
    se_W = np.sqrt(s2_W * np.diag(Sigma_W))

    # Compute t-values
    t_values = beta_W.flatten() / se_W
    
    #Compute R squared
    ss_total = np.sum((y_tilde - np.mean(y_tilde)) ** 2)
    ss_residual = np.sum((y_tilde - X_tilde @ beta_W) ** 2)
    R2 = 1 - (ss_residual / ss_total)
    
    return beta_W, t_values, se_W, R2, sigma2_heta
    
def main():
    file_path = "data/data_assignment1.csv"
    data = pd.read_csv(file_path)
    data = data.iloc[:, 2:]  # Ignore first two columns


    y = data[y_var].to_numpy()
    X = data[x_vars].to_numpy()

    beta_W, t_values, se_W, R2, sigma2_heta = within_regression(y,X)
    
    # Summary Table
    results = pd.DataFrame({
    "Coefficient": beta_W,
    "Std Error": se_W,
    "t-Statistic": t_values
    },  x_vars) 

    print(results)
    print(f"R-squared = {R2:.4f}")
    print(f"Estimated between regression variance = {sigma2_heta:.4f}")
    
    latex_output = results.style.to_latex()
    
    # Output Latex table
    with open("within_estimation_results.tex", "w") as f:
        f.write("\\begin{table}[h]\\centering\n")
        f.write(latex_output + "\n")
        f.write("\\caption{Within Estimation Results}\n")
        f.write("\\label{tab:within_estimation}\n")
        f.write("\\end{table}\n")
    
###########################################################
### call main
if __name__ == "__main__":
    main()
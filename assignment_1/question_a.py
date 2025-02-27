'''
Filename: question_a.py

Purpose: Carry out the between estimation and report the estimates, including standard errors and t-statistics,
but also the coefficient of determination R2 and the estimate of sigma_a^2
    
Date:
    27 February 2025  
'''

###########################################################
### Imports
import numpy as np
import pandas as pd

# global variables
y_var = "lwage"
x_vars = ["exp", "wks", "bluecol", "ind", "south", "smsa", "married", "gender", "union", "edu", "colour"]

###########################################################


def between_regression(y, X , N=595, T=7):
    I_T = np.ones((T, 1)) 
    I_N = np.eye(N) 
    D = np.kron(I_N, I_T)  
    P_D = D @ np.linalg.inv(D.T @ D) @ D.T  # Projection matrix  / Average maker
    
    # Averaged y and X per individual
    y_bar = (P_D @ y).reshape(N, T).mean(axis=1, keepdims=True)
    X_bar = (P_D @ X).reshape(N, T, X.shape[1]).mean(axis=1) 
    
    XtX_inv = np.linalg.inv(X_bar.T @ X_bar)
    beta_B = XtX_inv @ (X_bar.T @ y_bar)
    
    # Compute residuals
    a_i = y_bar - X_bar @ beta_B
    sigma2_alpha = np.sum(a_i**2) / (N - X.shape[1])
    
    # Compute standard errors
    s2_B = sigma2_alpha
    Sigma_B = XtX_inv
    se_B = np.sqrt(s2_B * np.diag(Sigma_B))

    # Compute t-values
    t_values = beta_B.flatten() / se_B
    
    #Compute R squared
    ss_total = np.sum((y_bar - np.mean(y_bar)) ** 2)
    ss_residual = np.sum(a_i**2)
    R2 = 1 - (ss_residual / ss_total)
    
    return beta_B, t_values, se_B, R2, sigma2_alpha
    
def main():
    file_path = "data/data_assignment1.csv"
    data = pd.read_csv(file_path)
    data = data.iloc[:, 2:]  # Ignore first two columns


    y = data[y_var].to_numpy().reshape(-1, 1)  # Ensure y is (NT, 1)
    X = np.column_stack((np.ones(len(y)), data[x_vars].to_numpy()))


    beta_B, t_values, se_B, R2, sigma2_alpha = between_regression(y,X)
    
    # Summary Table
    results = pd.DataFrame({
    "Coefficient": beta_B.flatten(),
    "Std Error": se_B,
    "t-Statistic": t_values
    }, index=["Intercept"] + x_vars)

    print(results)
    print(f"R-squared = {R2:.4f}")
    print(f"Estimated σ2_α  = {sigma2_alpha:.4f}")
    
    latex_output = results.style.to_latex()
    
    # Output Latex table
    with open("between_estimation_results.tex", "w") as f:
        f.write("\\begin{table}[h]\\centering\n")
        f.write(latex_output + "\n")
        f.write("\\caption{Between Estimation Results}\n")
        f.write("\\label{tab:between_estimation}\n")
        f.write("\\end{table}\n")
    
###########################################################
### call main
if __name__ == "__main__":
    main()
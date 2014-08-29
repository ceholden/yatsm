#include <iostream>
#include <stdio.h>
#include "armadillo"

using namespace std;
using namespace arma;

/*
 *  madsigma(vec r, int p)
 *  Description:
 *      returns sigma estimate using median absolute deviation (MAD) of 
 *      residuals  
 *  Input:
 *      Armadillo type vec of residuals
 *      int of rank of X
 */
double madsigma(vec r, int p) {
    // Sort residuals
    vec rs = sort(abs(r));
    // Get starting point for s
    int start = max(1, p);
    double s = median(rs.rows(start, rs.n_rows - 1)) / 0.6745;
    return (s);
}

/*
 * bisquare(vec r)
 * Decription:
 *      Returns weighting for each residual using bisquare weight function
 * Input:
 *      Armadillo type vec of residuals
 * Output:
 *      Armadillo type vec of weights
 *
 */
vec bisquare(vec r) {
    vec w = (abs(r) < 1) % pow((1 - pow(r, 2)), 2);
    return (w);
}

/*
 * find_rank(mat R, double tol)
 * Description:
 *      Finds rank of matrix R if R(i, i) > tol
 * Input:
 *      Armadillo type mat R
 *      double tolerance
 * Output:
 *      int rank of matrix R
 */

int find_rank(mat R, double tol) {
    int rank = 0;
    for (int i = 0; i < R.n_cols; i++) {
        if (R(i, i) > tol) {
            rank++;
        }
    }
    return(rank);
}

/*
 * weight_fit(vec y, mat X, vec w, int* rank)
 * Description:
 *      Takes X & Y and weights and returns betas for weighted OLS
 * Input:
 *      Armadillo type vec of Y
 *      Armadillo type mat of X
 *      Armadillo type vec of weights
 * Output:
 *      Armadillo type vec of betas
 */
vec weight_fit(vec y, mat X, vec w, int* rank) {
    vec sw = sqrt(w);
    vec yw = y % sw;
    mat Xw = X;
    for (int i = 0; i < X.n_rows; i++) {
        for (int j = 0; j < X.n_cols; j++) {
            Xw(i, j) = X(i, j) * sw(i);
        }
    }

    Xw.print("X weighted: ");

    // Do QR
    mat Q, R;
    qr(Q, R, Xw);

    *rank = find_rank(R, 1e-10);

    vec b;
    b = solve(R, (Q.t() * yw));
    return(b);
}

/*
 * check_any_diff(vec b0, vec b, double tol)
 * Description:
 *      Checks if difference between each element in b0 and b is less than tol.
 *      If any are smaller, returns true. Else returns false.
 */

bool check_any_diff(vec b0, vec b, double tol) {
    int n = b.n_rows;
    for (int i = 0; i < n; i++) {
        if (abs(b0(i) - b(i)) < tol) {
            return (true);
        }
        else {
            return (false);
        }
    }

}

template<int N>
vec robust_irwlstsq(double (&x)[N], double (&y)[N], float tune) {
    /*
     *
     * Why double(&x)[N]? See:
     * http://stackoverflow.com/questions/4839626/element-count-of-an-array-in-c
     *
     */

    cout << "x: " << x << endl;
    cout << "y: " << y << endl;

    // int n = sizeof(x) / sizeof(x[0]);
    int n = N;

    mat X = mat(n, 2);
    vec Y = vec(n);

    for (int i = 0; i < n; i++) {
        X(i, 0) = 1;
        X(i, 1) = x[i];
        Y(i) = y[i];
    }

    X.print("X: ");
    Y.print("Y: ");

    mat Q, R;
    qr(Q, R, X);

    // Get rank of X
    int rank = find_rank(R, 1e-10);

    printf("Rank is %i\n", rank + 1);

    R.print("R: ");
    Q.print("Q: ");

    // First go at solving LstSq
    vec b;
    b = solve(R, (Q.t() * Y));
    b.print("B0: ");

    // E = ?; h = leverage?
    mat E = X * R.submat(0, 0, rank, rank).i();
    E.print("E: ");

    vec h = vec(n);
    for (int i = 0; i < n; i++) {
        mat Esq = pow(E, 2);
        h(i) = sum(Esq.row(i));
        h(i) = min(0.9999, h(i));
    }
    h.print("h: ");

    // adjustment factor
    vec adjfact = vec(n);
    adjfact = 1 / sqrt(1 - h);
    adjfact.print("adj: ");

    double stdy = stddev(Y);
    printf("stddev y: %.5f\n", stdy);
    double tiny_s = stdy * 1e-6;
    if (tiny_s == 0) {
        tiny_s = 1;
    }
    printf("tiny_s: %g\n", tiny_s);

    // Do iteratively weighted LS
    vec b0 = b + 100; // vector to store betas from last iter
    float tol = 1e-10;
    int iter = 0;
    int iterlim = 50;
    double mad_s;
    vec weight;
    while (iter < iterlim && check_any_diff(b0, b, tol) == false) {
        cout << "<------------------------------ " << iter << endl;
        // Setup residuals
        vec resid = Y;
        // Loop through rows
        for (int i = 0; i < X.n_rows; i++) {
            // Loop through variables (columns)
            for (int j = 0; j < X.n_cols; j++) {
                // Subtract out each prediction by B[j]*X[i,j]
                resid(i) -= b(j) * X(i, j);
            }
        }
        // Adjust residuals
        vec radj = resid % adjfact;
        radj.print("ResidAdj:");
        mad_s = madsigma(radj, rank);
        cout << "MADsigma: " << mad_s << endl;

        vec rtune = radj / (max(mad_s, tiny_s) * tune);
        
        weight = bisquare(rtune);
        weight.print("Weight: ");

        // Keep old betas to check for convergence
        b0 = b;

        // Refit
        b = weight_fit(Y, X, weight, &rank);
        b.print("Beta:");

        iter++;
    }
    printf("madsigma: %.5f\n", mad_s);

    return(b);
}

int main() {

    /*
     * Example dataset from:
     * http://www.mathworks.com/help/stats/robustfit.html
     *
     */
    double x[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    double y[] = { 7.5, 4.1, 4.2, 2.1, -1.5, -1, -2.5, -6, -9, 0 };

    cout << "x: " << x << endl;
    cout << "y: " << y << endl;

    cout << "Armadillo iteratively reweighted least squares\n";
    float tune = 4.685;
    vec beta = robust_irwlstsq(x, y, tune);
    beta.print("\n\nFinal betas\n");
}

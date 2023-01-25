/*
The calling syntax is

   [x,qmin,parf,iters] = gqt(a,b,delta,rtol,itmax,par)

This is a gateway to (a modified) gqt.
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mex.h"
#include "matrix.h"

/* Input Arguments */

#define A      prhs[0]
#define B      prhs[1]
#define DELTA  prhs[2]
#define RTOL   prhs[3]
#define ITMAX  prhs[4]
#define PAR    prhs[5]

/* Output Arguments */

#define X      plhs[0]
#define QMIN   plhs[1]
#define PARF   plhs[2]
#define ITERS  plhs[3]

#define dgqt dgqt_

void dgqt() ;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int n, m, p;
  int info, its, itmax;
  double delta, par, rtol, atol = 1.0e-10;
  double *qmin, *wa1, *wa2, *x, *z, *a, *b, *parf, *iters;
  mxArray *WA1, *WA2, *Z;

  /* Check for proper number of arguments */

  if (nrhs != 6) mexErrMsgTxt("gqt requires 6 input arguments.");
  if (nlhs !=  4) mexErrMsgTxt("gqt requires 4 output argument.");

  /* Obtain input values */

  itmax = mxGetScalar(ITMAX);
  par = mxGetScalar(PAR);
  rtol = mxGetScalar(RTOL);
  delta = mxGetScalar(DELTA);

  m = mxGetM(A);  n = mxGetN(A);  p = mxGetM(B);
  if ((m != n) || (n != p)) mexErrMsgTxt("ERROR, dimensions don't agree.");

  /* Create storage for lhs arguments */

  X = mxCreateDoubleMatrix(n,1,mxREAL);
  QMIN = mxCreateDoubleMatrix(1,1,mxREAL);
  PARF = mxCreateDoubleMatrix(1,1,mxREAL);
  ITERS = mxCreateDoubleMatrix(1,1,mxREAL);

  /* Create local storage */

  Z = mxCreateDoubleMatrix(n,1,mxREAL);
  WA1 = mxCreateDoubleMatrix(n,1,mxREAL);
  WA2 = mxCreateDoubleMatrix(n,1,mxREAL);

  /* Assign pointers */

  a = mxGetPr(A);
  b = mxGetPr(B);
  x = mxGetPr(X);
  qmin = mxGetPr(QMIN);
  parf = mxGetPr(PARF);
  iters = mxGetPr(ITERS);

  z = mxGetPr(Z);
  wa1 = mxGetPr(WA1);
  wa2 = mxGetPr(WA2);

  dgqt(&n,a,&n,b,&delta,&rtol,&atol,&itmax,&par,qmin,x,&info,&its,z,wa1,wa2);

  *parf = par;
  *iters = its;

  /* Release local storage */

  mxDestroyArray(Z);
  mxDestroyArray(WA1);
  mxDestroyArray(WA2);
}

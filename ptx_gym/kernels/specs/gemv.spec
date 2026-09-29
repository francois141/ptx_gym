dim M;
dim K;

array a[M, K];
array x[K];
array y[M];

y[i] = sum(k in 0..K, a[i, k] * x[k]);

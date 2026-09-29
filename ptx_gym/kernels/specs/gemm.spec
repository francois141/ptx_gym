dim M;
dim N;
dim K;

array a[M, K];
array b[K, N];
array c[M, N];

c[i, j] = sum(k in 0..K, a[i, k] * b[k, j]);

// FlashAttention (NeurIPS 2022) forward pass, non-causal, with the
// softmax scale 1/sqrt(64) folded in. Written without the max
// subtraction the kernel carries: the two forms are equal over the
// reals, and this one stays inside the sums-of-exponentials fragment
// the equivalence check is complete for.

dim B;
dim H;
dim T;
dim D;

array q[B, H, T, D];
array k[B, H, T, D];
array v[B, H, T, D];
array o[B, H, T, D];

o[batch, head, row, feature] =
    sum(key in 0..T, exp(0.125 * sum(e in 0..D,
        q[batch, head, row, e] * k[batch, head, key, e])) *
        v[batch, head, key, feature]) /
    sum(key in 0..T, exp(0.125 * sum(e in 0..D,
        q[batch, head, row, e] * k[batch, head, key, e])));

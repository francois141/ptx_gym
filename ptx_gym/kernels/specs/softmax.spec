dim B;
dim D;

array input[B, D];
array output[B, D];

output[row, column] = exp(input[row, column] -
    max(k in 0..D, input[row, k])) /
    sum(k in 0..D, exp(input[row, k] - max(j in 0..D, input[row, j])));

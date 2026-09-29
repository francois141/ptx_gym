dim BATCH;
dim CIN;
dim IH;
dim IW;
dim COUT;
dim OH;
dim OW;
dim KH;
dim KW;

array x_ptr[BATCH, CIN, IH, IW];
array weight_ptr[COUT, CIN, KH, KW];
array output_ptr[BATCH, COUT, OH, OW];

output_ptr[b, co, ho, wo] = sum(ci in 0..CIN, sum(kh in 0..KH, sum(kw in 0..KW,
  x_ptr[b, ci, ho + kh, wo + kw] * weight_ptr[co, ci, kh, kw])));

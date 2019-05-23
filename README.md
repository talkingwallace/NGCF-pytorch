# GraphNN
A pytorch implementation of Neural Graph Collaborative filtering
link:https://arxiv.org/pdf/1905.08108.pdf
Tested on toy dataset movielens 100k

# Details
Add three transform layer to yield prediction of ratings

# Evaluation
Train 0.9 test 0.2
SVD dim 50 RMSE 0.931
NCF dim 64 layers [128,64,32,8] RMSE 0.928
NGCF dim 64 layers [64,64,64] RMSE 0.914


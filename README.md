# SVD++



Part I

1. Rating prediction formula and its explanation 

Let  $R_{n*m}$ be a rating matrix containing the ratings of $n$ users for $m$  items. Each matrix element  $r_{ui}$ refers to the rating of user $u$ for item  $i$. 

The predictive rating of the SVD++ model is
$$
r_{ui} = \mu + b_u + b_i + q_i^T \left(p_u + \frac{1}{\sqrt{|R(u)|}}\sum_{j\in R(u)} y_j \right)
$$

where $μ$ is the overall average rating and $b_u$ and $b_i$ indicate the observed deviations of user $u$ and item $i$, respectively. $R(u)$ is the set of items rated by user $u$, $y_j$ represents the implicit feedback vector of item j.

2. Objective function

$$
\begin{aligned}
& \sum_{r_{u i} \in R}\left[r_{u i}-\mu-b_{u}-b_{i}-q_{i}^{T} \cdot\left(p_{u}+|R(u)|^{-1 / 2} \sum_{j \in R(u)} y_{j}\right)\right. \\
&\left.\quad+\lambda_1\left(b_{u}^{2}+b_{i}^{2}\right)+\lambda_2\left(\left\|p_{u}\right\|^{2}+\left\|q_{i}\right\|^{2}\right)\right]
\end{aligned}
$$

Besides error between estimate rating and actual rating, regularization was introduced in order to avoid overfitting. It is penalty on the parameter, make sure it will not become to large to affact the result dominantly. I think the reason why a regularization is necessary in this case is that the data is really sparse compared to the parameters in model. After several iterations, the model is very likely to 'memorize' all the ratings. Thus, the loss on the training set will not match the loss on the validation set.  In accordance to the paper, $\lambda_1$is set to 0.005, $\lambda_2$ is set to 0.015 in my code.

3. Parameter update rules by SGD

   for each batch, stochastically choose data, and update $b_u$, $b_i$, $q_i$, $p_u$ and $y_j$ according to following rules:

   - $b_{u} \leftarrow b_{u}+\gamma \cdot\left(e_{u i}-\lambda_{1} \cdot b_{u}\right)$

   - $b_{i} \leftarrow b_{i}+\gamma \cdot\left(e_{u i}-\lambda_{1} \cdot b_{i}\right)$

   - $q_{i} \leftarrow q_{i}+\gamma \cdot\left(e_{u i} \cdot\left(p_{u}+|\mathrm{R}(u)|^{-\frac{1}{2}} \sum_{j \in \mathrm{R}(u)} y_{j}\right)-\lambda_{2} \cdot q_{i}\right)$

   - $p_{u} \leftarrow p_{u}+\gamma \cdot\left(e_{u i} \cdot q_{i}-\lambda_{2} \cdot p_{u}\right)$

   - $\forall_{j} \in \mathrm{R}(u): y_{j} \leftarrow y_{j}+\gamma \cdot\left(e_{u i} \cdot|\mathrm{R}(u)|^{-\frac{1}{2}} \cdot q_{i}-\lambda_{2} \cdot y_{j}\right)$


4. Relations with other latent factor models 

The SVD++ model, which is a derivative model of SVD, is the research object, and three new algorithms that apply DP to SVD++ using gradient perturbation, objective-function perturbation, and output perturbation are proposed. To improve the predictive accuracy, SVD++ considers the related information of the user and item. The theoretical proofs are given and the experiment results show that the new private SVD++ algorithms obtain better predictive accuracy, compared with the same DP treatment of traditional MF and SVD. The DP parameter is the key to the privacy protection power, but in the current study, it was selected by experience. Finally, an effective trade-off scheme is given that can balance the privacy protection and the predictive accuracy to a certain extent and can provide a reasonable range for parameter selection. 

5. Pseudo-code Algorithm

   ```
   SVD++ Algorithm:
   Input:  m       # numbers of users
           n       # numbers of items
           k       # the length of p & q, hyper-parameter
           p_u     # vector of user preference
           q_i     # vector of item quality
           b_i     # item bias
           b_u     # user bias
           y_j     # implicit feed back on item j
           epochs  # total epochs
           lr      # learning rate
           decay   # decay of learning rate
           l1      # regularization parameter1
           l2      # regularization parameter2
           ts      # training set
           rui     # error of the u_th user on i-th item
   initialize all the vectors, both for users and items
   for epoch in epochs: # in each epoch:
       for data in training set:
           calculate rui;
           update weights in vectors:
               b_u = b_u + lr * (eui - l1 * b_u)
               b_i = b_i + lr * (eui - l1 * b_i)
               q_i = q_i + lr * (eui * (p_u + 1 / (len(R(U)) ** 0.5) * sigmaYj(u)) - l2 * q_i)
               p_u = p_u + lr * (eui * q_i - l2 * p_u)
               for j in R(u):
                   y_j = y_j + lr * (eui * 1 / (len(R(u)) ** 0.5) * q_i - l2 * y_j)
       print training time, calculate MAE and RMSE loss;
       lr = lr * decay
   
   
   
   ```

Part  II

1. Results including MAE/RMSE/Training Time/Test Time by 5-fold cross validation

   I use 50 as the value of K, which is the length of $p_u$ and $q_i$. And I set other hyper parameters in accordance with the SVD++ paper. My result is shown below.

​	total training time	=	19837.807s		average training time for one epoch		 = 	132.252s
​	total test time			=	1418.657s		  average test time for MAE and RMSE	  = 	9.458s

​	SVD++:

​	MAE loss				   =    0.74611376		RMSE loss 													= 	0.94432803	

​	Baseline:

​	MAE loss				   =    0.75681015		RMSE loss 													= 	0.95814728	

​		

2. The curve of loss value relative to training iterations 

​	![image-20220331125215439](https://tva1.sinaimg.cn/large/e6c9d24egy1h1dp1r3qrsj20hs0dct9y.jpg)


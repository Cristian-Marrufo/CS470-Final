# CS470-Final
## Final Project: Email Spam Classification

#Instructions:
- Implement the Naive Bayes, logistic regression, and KNN algorithm models for the purposes of Email spam classification given the provided data set.
- Evaluate the performance of each model given the indidual instructions for each.

#Naive bayes

1. model learning:

   Note:

   features: remove attributes that is not related to word (the last four attributes)

   labels: the last column

   count P(c) -> how many samples are positive, and how many are negtive

   if freq_word>0, then this word exists. You could use this to calculate P(a|c) -> for each class, what is the prob of each word

   remember to use laplace smoothing.

2. model evaluation (on val dataset -> performance(model, val)):
   
   for each new sample, $\prod{P(a|c)}P(c)$ if word is in the email(freq_word > 0); and find the maximum class


# KNN
1. model learning: None

2. model evaluation(on val dataset): You could use each row(exclude the last column) as the feature of the email. You do not have to recalcuate the freqency.

   ```
   Note:
   parallel programing
   numpy.cos() to calcuate the similarity
   ```


# LR

1. model learning: You could use each row(exclude the last column) as the feature of the email. You do not have to recalcuate the freqency.
    
    $y = sigmoid(MX)$

step 1: add one more column (all value is 1) in X -> X' = np.c_[np.ones((len(X), 1)), X]

step 2:vector M = np.random.randn(len(X[0])+1, 1);

key formula for step 3 (Note: n is the size of the TRAINING dataset; $cdot$ is dot production ):

1. $pred_y = sigmoid(M\cdot X')$

2. $loss = -\sum(y\cdot log(pred_y)+(1-y)\cdot log(1-pred_y))/n$

3. $gm=X'\cdot (pred_y - y)*2/n$

Step 3 example code:
   ```
   #Step 3: performing gradient descent on whole dataset:
   best_model = M
   best_performace = 0
   for i in range(epoch):
     pred_y = ...
     gm = ...
     _p = performace(model, val)
     if _p > best_performance:
        best_model = M
        best_performance = _p
     M = M - learning_rate*gm
   ```

2. model evaluation(on val dataset):
  
   calculate pred_y, if more than 0.5, then the predicted label is 1.

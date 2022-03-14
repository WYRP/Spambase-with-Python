from knnclassifier import knnclassifier
from classifier import data_item, normalize_dataset
from random import shuffle
from extra_trees import extra_tree


# I have provided this code to read the data from the file.
fp = open('spambase.data')
dataset = []
for line in fp:
    fields = line.split(',')
    data = [float(x) for x in fields[:-1]] # convert to float
    label = int(fields[-1]) # final item is the label
    dataset.append(data_item(label, data))

print("Read {} items.".format(len(dataset)))
# The number of features is just the length of a feature vector.
print("{} features per item.".format(len(dataset[0].data)))

# YOUR CODE GOES AFTER THIS LINE!

def evaluate(dataset, cls, n_folds = 0, **kwargs):
    if n_folds == 0:
        n_folds = len(dataset)
    n_features = len(dataset[0].data)
    test_size = round(len(dataset) / n_folds)
    index = 0
    global TN          
    global actual_good 
    global TP 
    global actual_spam 
    
    TN = 0           
    actual_good = 0
    TP = 0
    actual_spam = 0
    
    n_correct = 0           
    n_tested = 0
    
    
    for fold in range(n_folds):
        shuffle(dataset)
        train_data = dataset[:index] + dataset[index + test_size:]
        test_data = dataset[index:index + test_size]
        p = cls(**kwargs) 
        p.train(train_data)
        
        for item in test_data:
            if item.label==0:
                if p.predict(item.data)==0:
                    TN+=1
                actual_good+=1
            elif item.label==1:
                if p.predict(item.data)==1:
                    TP+=1
                actual_spam+=1
         
        index += test_size
    print("# of data in traning set:", len(train_data), ", # of data in testing set:", len(test_data))
    return "predic_good:", TN, "actual_good:", actual_good, "predict_spam:", TP, "actual_spam:", actual_spam

print(evaluate(dataset, knnclassifier, 15, K=15))

FN=actual_good-TN
FP=actual_spam-TP

#3
confusion_matrix=print('{:6s}{}'.format("0", "1"), "\n", "0", TN, FN,
                       "\n", '{:1s}'.format("1"), FP, TP)


#4:
print("TPR=", TP/(TP+FN), "FPR=", FP/(FP+TN))










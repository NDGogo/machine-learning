# H2O

## Requirements
    - H2O recommends the cluster size to be four times the size of the data.

## Concepts
    - All the data is on the cluster (server), not on the client. Even when client and cluster are the same machine. Therefore to train a model, or make a preduction, we have to get the data into the H2O cluster. (Cook 2016, p 25)
    - Every change made involves a data copy. That means the frame will change too. A lot of operations in H2O are lazy, meaning the requested change is recorded but not carried out until it has to be. (Cook 2016, p 34)

## Code Snippets
    - 
 ```python
    # Splitting data frames: the 0.8 tells H2O to put 80% in the first split, the rest in the second split.
    train, test = data.split_frame([**0.8**])
```
```python
    # Generating confusion matrix:
    model.confusion_matrix(train)
```
    
```python
    # Making predictions:
    model.predict(test)
```     

## Reference List
    - Cook, D. 2016, 'Practical Machine Learning with H2O', 1st edn.

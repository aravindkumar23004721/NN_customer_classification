# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="625" alt="Screenshot 2025-03-21 at 9 32 27 AM" src="https://github.com/user-attachments/assets/4b1a57eb-89f1-4897-b674-ff3e6093f17e" />

## DESIGN STEPS

### Step 1: 
Import necessary libraries and load the dataset.

### Step 2: 
Encode categorical variables and normalize numerical features.

### Step 3: 
Split the dataset into training and testing subsets.

### Step 4: 
Design a multi-layer neural network with appropriate activation functions.

### Step 5: 
Train the model using an optimizer and loss function.

### Step 6: 
Evaluate the model and generate a confusion matrix.

### Step 7: 
Use the trained model to classify new data samples.

### Step 8: 
Display the confusion matrix, classification report, and predictions.

## PROGRAM

### Name:Aravind Kumar SS
### Register Number: 212223110004

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

```
```python
def train_model(model,train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      outputs=model(X_batch)
      loss=criterion(outputs,y_batch)
      loss.backward()
      optimizer.step()

  if(epoch+1)%10==0:
    print(f'Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}')

```

## Dataset Information

![Screenshot 2025-03-24 113802](https://github.com/user-attachments/assets/ccc228bc-d022-4ef6-a5ae-d89578462000)

## OUTPUT

### Confusion Matrix

![Screenshot 2025-03-24 113830](https://github.com/user-attachments/assets/7b40d766-ce73-4b2a-a023-7f294100b2bf)

### Classification Report

![Screenshot 2025-03-24 113819](https://github.com/user-attachments/assets/93182fbb-6c1f-450f-ae72-f0d5fb5cb0d8)


### New Sample Data Prediction

![Screenshot 2025-03-24 113839](https://github.com/user-attachments/assets/64a2b02f-87d4-405a-ac83-b48953974442)

## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.

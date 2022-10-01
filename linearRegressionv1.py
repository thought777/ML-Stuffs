import numpy as np
import pandas as pd


#Most basic linear regression

def calculate_Val_x(xValue):
    return 2*xValue+3

def generate_Data_Set(no_Of_rows=50):
    dataFrame=pd.DataFrame(columns=['x'])
    dataFrame.loc[:,'x']=np.arange(no_Of_rows)
    dataFrame["y"]=dataFrame.apply(lambda row:calculate_Val_x(row),axis=1)
    # print(dataFrame)
    return dataFrame

def hypothesis(theta,xVal):
    # row,col=theta.shape
    # print(theta[0]+theta[1]*xVal)
    # print(theta)
    return theta[0]+theta[1]*xVal

# def costFunction(dataArray,theta):
#     # cost=0
    
#     return tempSum*0.5

def batch_Gradient_Descent(trainDtaframe):
    n,j=trainDtaframe.shape
    # print("n:",n,"j:",j)
    theta=np.random.rand(j)
    # print('-------------')
    # print(theta)
    dataArray=trainDtaframe.to_numpy()
    # print(dataArray)
    r,c=dataArray.shape
    # print(f'no. of records:{r}')
    alpha=0.00001
    while True:
        tempSum=0
        for i in range(r):
            x=dataArray[i][0]
            # print(f'x:{x}')
            # print(f'h(x):{hypothesis(theta, x)}')
            # print(f'y:{dataArray[i][1]}')
            tempSum+=(dataArray[i][1]-hypothesis(theta, x))*x
        # print(f'Error:{tempSum}')
        if tempSum<0.0000001:
            break
        # print(theta.shape)
        for j in range(theta.shape[0]):
            theta[j]+=tempSum*alpha
        # alpha*=0.001
    print(f'Error:{tempSum} and alpha={alpha}')
    return theta



def main():
    data=generate_Data_Set()
    print(batch_Gradient_Descent(data))

if __name__ == "__main__":
    main()
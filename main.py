# //////////////////////////////
# ////////////    packages      ////////////////
# //////////////////////////////
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from IPython.core.pylabtools import figsize
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from IPython.display import clear_output

df = pd.read_csv('theCars.csv')

# ///////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////       declare functions       //////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////
# //////////    q1           /////////////////
# /////////////////////////////
def readData():
    print(df)

# /////////////////////////////
# /////////     q2            /////////////////
# ////////////////////////////
def chartCo2PVolume():
    Volume = df['Volume']
    CO2 = df['CO2']
    plt.scatter(Volume, CO2, color="#004225")
    plt.xlabel("Volume")
    plt.ylabel("CO2")
    plt.title("CO2 / Volume")
    plt.show()

# /////////////////////////////
# /////////     q3            /////////////////
# ////////////////////////////
def chartCo2PWeight():
    Weight = df['Weight']
    CO2 = df['CO2']
    plt.scatter(Weight, CO2, color="#004225")
    plt.xlabel("Weight")
    plt.ylabel("CO2")
    plt.title("CO2 / Weight")
    plt.show()

# ///////////////////////
# ///////       q4      ////////////////
# ///////////////////////
def chartVolumeWeightCo2():
    Volume = df['Volume']
    Weight = df['Weight']
    Co2 = df['CO2']
    ax = plt.axes(projection='3d')
    ax.scatter(Volume, Weight, Co2, color='red', s=100)
    ax.set_xlabel('volume')
    ax.set_ylabel('weight')
    ax.set_zlabel('co2')
    plt.show()

# ///////////////////////
# ////////      q5      ///////////////////
# ///////////////////////
def findTestTrain():
    test = df[((df['Car'] == 'Opel') & (df['Model'] == 'Astra')) | ((df['Car'] == 'Mazda') & (df['Model'] == '3')) | (
                (df['Car'] == 'Ford') & (df['Model'] == 'Focus')) | (
                          (df['Car'] == 'Suzuki') & (df['Model'] == 'Swift')) | (
                          (df['Car'] == 'Hyundai') & (df['Model'] == 'I20')) | (
                          (df['Car'] == 'Fiat') & (df['Model'] == '500'))]
    test = test.reset_index()
    temp = df.reset_index()
    print(test)
    train = temp[temp['index'].isin(test['index'])==False]
    return {'train':train,'test':test}

# ////////////////////////
# ///////       q6      ///////////////////
# ////////////////////////
def findTheta01(dataset,column):
    meanVOL = np.mean(dataset[column])
    stdVOL = np.std(dataset[column])
    x = ((dataset[column] - meanVOL) / stdVOL)
    x = x.values.reshape(-1,1)
    meanCO2 = np.mean(dataset['CO2'])
    stdCO2 = np.std(dataset['CO2'])
    y = ((dataset['CO2'] - meanCO2) / stdCO2)
    y = y.values.reshape(-1,1)
    model = LinearRegression().fit(x,y)
    theta0 = model.intercept_
    theta1 = model.coef_
    print("theta0 = ",theta0)
    print('theta1 = ',theta1)
    xspace = np.linspace(min(x),max(x))
    yspace = model.predict(xspace)
    plt.plot(xspace,yspace,color="black",linewidth=3)
    plt.scatter(x,y)
    plt.show()
# ////////////////////////
# ////////      q7      ////////////////////
# ///////////////////////
def findRSQ(dataset,column):
    xTrain = dataset['train'][column].values.reshape(-1,1)
    yTrain = dataset['train']['CO2'].values.reshape(-1,1)
    model = LinearRegression().fit(xTrain,yTrain)
    R_SQTrain = model.score(xTrain,yTrain)
    xTest = dataset['test'][column].values.reshape(-1,1)
    yTest = dataset['test']['CO2'].values.reshape(-1,1)
    model = LinearRegression().fit(xTest,yTest)
    R_SQTest = model.score(xTest,yTest)
    print("R_SQ test= ",R_SQTest)
    print("Dif R_SQ= ",abs(R_SQTrain-R_SQTest))
# //////////////////////////////////
# ////////////      q8            ////////////////////
# //////////////////////////////////
def findAlpha(column):
    def featureNormalization():
        """
        Take in numpy array of X values and return normalize X values,
        the mean and standard deviation of each feature
        """
        meanVOLCO2 = np.mean(df[['Volume','CO2']],axis=0)
        stdVOLCO2 = np.std(df[['Volume','CO2']], axis=0)

        X_norm = (df[['Volume','CO2']]-meanVOLCO2)/stdVOLCO2

        return X_norm
    def computeCost(X, y, theta):
        """
        Take in a numpy array X,y, theta and generate the cost function of using theta as parameter
        in a linear regression model
        """
        m = len(y)
        predictions = X.dot(theta)
        square_err = (predictions - y) ** 2

        return (1 / (2 * m) * np.sum(square_err))
    def gradientDescent(X, y, theta, alpha, num_iters, data,column):
        """
        Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
        with learning rate of alpha
        return theta and the list of the cost of theta during each iteration
        """
        m = len(y)
        J_history = []
        fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
        axes[0].scatter(data[column], data['CO2'])
        for i in range(num_iters):
            predictions = X.dot(theta)
            error = np.dot(X.transpose(), (predictions - y))
            descent = alpha * 1 / m * error
            theta -= descent
            J_history.append(computeCost(X, y, theta))

            axes[2].plot(J_history)
            x_value = [x for x in range(500,2800)]
            y_value = [y * theta[1] + theta[0] for y in x_value]
            axes[1].plot(x_value, y_value, color="r")
            axes[1].set_xlabel(column)
            axes[0].set_ylabel("CO2")
            axes[2].set_xlabel("Iteration")
            axes[2].set_ylabel("$J(\Theta)$")
            plt.pause(0.011)
            clear_output(wait=True)
        return theta, J_history


    Xy = featureNormalization()
    X = np.append(np.ones((36,1)),Xy['Volume'].values.reshape(36,1),axis=1)
    y = Xy['CO2'].values.reshape(36,1)
    theta = np.random.rand(2, 1)
    data = df[['Volume','CO2']]
    theta,J_history = gradientDescent(X,y,theta,0.0199,100,data,column)
    print("h(x) =" + str(np.round(theta[0,0],4)) + " + " + str(np.round(theta[1, 0], 4)) + "x1")

# ////////////////////////////////
# ////////////      q9          ////////////////////
# ////////////////////////////////
def question9(dataset,column):
    findTheta01(dataset,column)
    findRSQ(findTestTrain(),column)
    findAlpha(column)
# ///////////////////////////////
# ///////////////    q10   ////////////////////
# ///////////////////////////////
def findRSQMultiReg(dataset):
    x_train = dataset['train'][['Volume','Weight']]
    y_train = dataset['train']['CO2']
    model = LinearRegression().fit(x_train,y_train)
    x_test = dataset['test'][['Volume','Weight']]
    y_test = dataset['test']['CO2']
    R_SQTest = model.score(x_test,y_test)
    print('R_SQTest = ',R_SQTest)
    return model
# //////////////////////////////////////
# ////////////          q11     ///////////////////////
# /////////////////////////////////////
def findCarCatByMinRSQ(model):
    Y_predict = []
    difError = []
    temp = df.reset_index()
    X = temp[['Volume','Weight']].values.reshape(-1,2)
    y = temp['CO2'].values.reshape(-1,1)
    Y_predict.append(model.predict(X))
    for i in range(0,36):
        difError.append(np.power(y[i][0]-Y_predict[0][i],2)/36)
    # print('min Error =',np.amin(difError, axis=0))
    index = np.where(difError==np.amin(difError,axis=0))
    print(temp[temp['index']==index[0][0]+1][['Car','Model']])
# /////////////////////////////////////
#//////////////         q12         //////////////////////////
# /////////////////////////////////////
def modelClassification(dataset):
    model = LogisticRegression().fit(dataset['train'][['Volume','Weight','CO2']],dataset['train']['Car'])
    print("classes =",model.classes_)
    print("intercept =",model.intercept_)
    print("coef = ",model.coef_)
    return model
# ////////////////////////////////////////
# ////////////          q13        ////////////////////////////
# ///////////////////////////////////////////////////
def findAccuracyTrain(model,dataset):
    print("score =",model.score(dataset['train'][['Volume','Weight','CO2']],dataset['train']['Car']))
    print("accuracy score =",metrics.accuracy_score(dataset['train']['Car'],model.predict(dataset['train'][['Volume','Weight','CO2']])))
# ////////////////////////////////////////
# ////////////          q14        ////////////////////////////
# ///////////////////////////////////////////////////
def findAccuracyTest(model,dataset):
    print("score =",model.score(dataset['test'][['Volume','Weight','CO2']],dataset['test']['Car']))
    print("accuracy score =",metrics.accuracy_score(dataset['test']['Car'],model.predict(dataset['test'][['Volume','Weight','CO2']])))
def findMaxAccuracy(model):
    print("accuracy =",metrics.accuracy_score(df[df['Car']=='Skoda']['Car'],model.predict(df[df['Car']=='Skoda'][['Volume','Weight','CO2']])))
    print("Skoda")
# /////////////////////////////////////////////////////////////////////////////////
# ///////////////////           Run functions           //////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////
# q1
# readData()
# q2
# chartCo2PVolume()
# q3
# chartCo2PWeight()
# q4
# chartVolumeWeightCo2()
# q5
# findTestTrain()
# q6
# findTheta01(findTestTrain()['train'],'Volume')
# q7
# findRSQ(findTestTrain(),'Volume')
# q8
findAlpha('Volume')
# q9
# question9(findTestTrain()['train'],'Weight')
# q10
# findRSQMultiReg(findTestTrain())
# q11
# findCarCatByMinRSQ(findRSQMultiReg(findTestTrain()))
# q12
# modelClassification(findTestTrain())
# q13
# findAccuracyTrain(modelClassification(findTestTrain()),findTestTrain())
# q14
# findAccuracyTest(modelClassification(findTestTrain()),findTestTrain())
# q15
# findMaxAccuracy(modelClassification(findTestTrain()))

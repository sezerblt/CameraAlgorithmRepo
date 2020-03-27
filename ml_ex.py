import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
import mysql.connector

#-----------StDev Class--------------------------|
class StandartDeviation():
        
    def __init__(self,data):
        self.data=data

    def getStandartDeviation(self):
        sigma=np.std(self.data)
        return sigma

    def getVariance(self):
        sigma_square=np.var(self.data)
        return sigma_square

    def getPercentile(self):
        perc=np.percentile(self.data,60)
        return perc

    def showHistogram(self):
        plt.hist(self.data,100)
        plt.show()

    def normalDistrubition(self):
        plt.hist(self.data,100)
        plt.show()

    def scatterPlot(self):
        x=np.random.normal(0.0,10.0,100)
        y=np.random.normal(0.0,10.0,100)
        plt.scatter(x,y)
        plt.show()
#-------------------------------------------------|

#-----------ML Test Class-------------------------|
class MachineLearning():

    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.slope=None
        self.intercept=None

    def process(self,x):
        return self.slope*x + self.intercept

    def regression(self):
        self.slope,self.intercept,coeff,pval,std_error=stats.linregress(self.x,self.y)
        print("slope: ",self.slope)
        print("intercept: ",self.intercept)
        print("coefficient: ",coeff)
        print("p value: ",pval)
        print("standart error: ",std_error)
        our_model=list(map(self.process,self.x))
        plt.scatter(self.x,self.y)
        plt.plot(self.x,our_model)
        plt.show()

    def polynomialRegression(self):
        model=np.poly1d(np.polyfit(self.x,self.y,3))
        print("model: ",model)
        myline=np.linspace(0,50,100)
        print("line: ",myline)
        plt.scatter(self.x,self.y)
        plt.plot(myline,model(myline))
        plt.show()

    def predictFuture(self,predict):
        mymodel = np.poly1d(np.polyfit(x, y, 3))
        speed = mymodel(predict)
        print("estimate of speed: ",speed)
#--------------------------------------------------------

#-----------AbstractClass Class-------------------------- 
class AbstractAction:
    
    @abstractmethod
    def productCar(self):pass
    
    @abstractmethod
    def destroyCar(self):pass
#--------------------------------------------------------

#-----------Vehicle Class-------------------------------- 
class Vehicle:
    "Vehicles"
    def __init__(self,mark):
        self.mark_mark=mark
        self.gear=0

    def overriddingMethod(self):
        return "Vehicle Class Method"

    def increaseGear(self):
        if self.gear<0:
            self.gear=gear
        else:
            self.gear+=1

    def reductionGear(self):
        if self.gear<0:
            self.gear=gear
        else:
            self.gear-=1
#----------------------------------------------------

#-----------CAR Class--------------------------------    
class Car(Vehicle,AbstractAction):
    carList=None
    "Cars"
    def __init__(self,mark,model,volume,weight):
        Vehicle.__init__(self,mark)
        self.__model= model
        self.__volume = volume
        self.__weight = weight
        self.gear=0
        

    def getModel(self):
        return self.__model
    def setModel(self,model):
        self.__model=model

    def getVolume(self):
        return self.__volume
    def setVolume(self,volume):
        self.__volume=volume

    def getWeight(self):
        return self.__weight
    def setWeight(self,weight):
        self.__weight=weight

    def productCar(self):
        return "New Car created"
    
    def destroyCar(self):
        return "Destroyed the Car"

    def overriddingMethod(self):
        return "Car Class Method"

    def increaseGear(self):
        if self.gear<0:
            self.gear=gear
        else:
            self.gear+=2

    def reductionGear(self):
        if self.gear<0:
            self.gear=gear
        else:
            self.gear-=2

    @classmethod
    def connectDataBase(cls):
        cls.mydb = mysql.connector.connect(
                                  user="root",
                                  password="password",
                                  host="127.0.0.1",
                                  database="company"
        )
        print(cls.mydb)
        return cls.mydb
        """
        crsr=mydb.cursor()
        query=("Select * from car")
        crs.execute(query)
        return crs
        """

    @classmethod
    def getCarList(cls):
        db=cls.connectDatabase()
        crsr=db.cursor()
        query=("Select * from car")
        cls.carList=crsr.execute(query)
        return cls.carList
        
#-------------------------------------------
class Student():
    
    def __init__(self,stud_id,stud_name,stud_surname):
        self.student_id  =stud_id
        self.student_name=stud_name
        self.student_surname=stud_surname
        self.db=None

    def connectDatabase(self):
        self.db=mysql.connector.connect(
                user="root",
                password="password",
                host="127.0.0.1",
                database="company"
            )
        return self.db
    

    def createData(self,statement):
        data=self.connectDatabase()
        mycursor=data.cursor()
        mycursor.execute(statement)

    def dropData(self,statement):
        data=self.connectDatabase()
        mycursor=data.cursor()
        mycursor.execute(statement)

    def showDatabase(self,statement):
        data=self.connectDatabase()
        mycursor=data.cursor()
        mycursor.execute(statement)
        for i in mycursor:
            print(i)

    def insertData(self,statement,value):
        data=self.connectDatabase()
        mycursor=data.cursor()
        
        mycursor.execute(statement,value)
        data.commit()
        print(mycursor.rowcount," was inserted")
        print("recorded :",mycursor.lastrowid)

    def deleteData(self,sql):
        data=self.connectDatabase()
        mycursor=data.cursor()
        
        mycursor.execute(sql)
        data.commit()
        print(mycursor.rowcount," deleted")

    def updateData(self,sql):
        data=self.connectDatabase()
        mycursor=data.cursor()
        
        mycursor.execute(sql)
        data.commit()
        print(mycursor.rowcount," updated")

    def getAllStudentList(self):
        data=self.connectDatabase()
        mycursor=data.cursor()
        mycursor.execute("select * from student")
        all_list=mycursor.fetchall()
        return all_list

    def getResultFilter(self,sql):
        data=self.connectDatabase()
        mycursor=data.cursor()
        mycursor.execute(sql)
        result_list=mycursor.fetchall()
        return result_list

#a=Student(15,"semsettin","canikli")
#a.showDatabase("show databases")
#statement="insert into  student (student_id,student_name,student_surname) values (%s,%s,%s)"
#value=(a.student_id,a.student_name,a.student_surname)
#a.insertData(statement,value)
#list_stud=a.getAllStudentList()
#for i in list_stud:
#    print(i)

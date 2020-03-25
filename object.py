import datetime
import json
import re,os
import numpy as np

class Person:
    def __init__(self, first_name, last_name):
        self.firstname = first_name
        self.lastname = last_name

    def printname(self):
      print(self.firstname, self.lastname)

class Student(Person):
    def __init__(self, first_name, last_name,year=datetime.datetime.now()):
        super().__init__(first_name, last_name)
        self.graduated_year=year

    def aboutStudent(self):
        print(self.firstname,' -',
              self.lastname,' Graduated::',self.graduated_year)

    def lessonList(self):
        lessons=('math','geo','eng','hist','art','music','comp')
        #iter_les = iter(lessons)
        #print(next(iter_les))
        for i in lessons:
            print(i)
    

    def __iter__(self):
        self.count=1
        return self
    
    def __next__(self):
        if self.count < 5:
            next_item = self.count
            self .count += 1
            return next_item
        else:
            raise StopIteration

    def setScore(self):
        self.average=0
        self.total=0
        for i in range(3):
            self.score=int(input("exam entry:"))
            self.total=self.total+self.score 
        self.average = int(self.total/3)
        
        if not type(self.average) is int:
            raise TypeError('not greater to a hundred')
        if self.average<0 and self.average>100:
            raise Exception('not negative and self.score>100')
        else:
            try:
                if self.average>=50:
                    print("you pass note:{:.2f}".format(self.average))
                else:
                    print("dont pass,note:{:.2f}".format(self.averae))
                    
            except:
                pass    

    def worktime(self):
        self.thetime=datetime.datetime.now()
        return self.thetime,':',self.thetime.strftime('%A')

    def testExp(self):
        self.text = "this is text function on sample"
        if(re.search("^this.*sample$",self.text)):
            print("yeahh this match")
        else:
            print("not match")

#ali=Student('ali','ak',datetime.datetime(2020,5,5,11,11,15))
#ali.aboutStudent()
#print(ali.worktime())
#ali.testExp()
#ali.setScore()

stu='{"first_name":"sedat","last_name":"kus","year":"2020-10-10"}'
res=json.loads(stu)
#print(res["first_name"])
#print(res["last_name"])
#print(res["year"])

x={"country":"turkey",
   "language":"turkish",
   "code":90,
   "inEuropa":True,
   "parts":("ege","marmara","karadeniz"),
   "some_cities":[
       {"name":"istanbul","code":34,"population":20000000},
       {"name":"ankara",  "code":6, "population": 6000000},
       {"name":"antalya", "code":7, "population": 2000000}
       ]
   ,"capital":"ankara"
   }
#jsonFormat=json.dumps(x,indent=6,separators=(",",":"),sort_keys=True)
#print(jsonFormat)
#--------------------------------------------------
class Country:
    code=0
    
    def __init__(self,country_name,code):
        self.name=country_name
        self.code=code

    def info(self):
        print(self.name+' is very beauty country..')

    def allList(self):
        self.file_list = open("country_list.txt","r")
        #read of all file
        #print(self.file_list.read())
        
        for x in self.file_list:
            #reading file on the line by line
            print(x)
        self.file_list.close()

    def writeToFile(self,input_text):
        self.file_list = open("country_list.txt","a")
        self.file_list.write(input_text)
        self.file_list.write("\n")
        self.file_list.close()
        
        self.file_list =open("country_list.txt","r")
        print(self.file_list.read())
        self.file_list.close()

class City(Country):
    code=1

    def __init__(self,country_name,code,city_name,zip_code):
        Country.__init__(self,country_name,code)
        self.city_name = city_name
        self.zip_code = zip_code

    def infoCity(self):
        print(self.zip_code,':',self.city_name)

    def createCityFile(self):
        #create a new file if it does not exist: 
        self.filex=open("cityFile.txt","w")

    def deleteFile(self,filename):
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print("The file does not exist")
    def deleteFolder(self,foldername):
        if os.path.exists(foldername):
            os.rmdir(foldername)
        else:
            print("The folder does not remove")
#---------------------------------------------
#chile=Country("Chile Republic",154)
#chile.writeToFile(chile.name)

import scipy
class Car:
    def __init__(self,
                 car_name='Demo',
                 color='white',
                 product_year=datetime.datetime.now(),
                 max_speed=100,
                 autoPass=False):
        self.car_name=car_name
        self.color=color
        self.product_year=product_year.year
        self.max_speed=max_speed
        self.autoPass=autoPass
        
class Node():
    def __init__(self,data):
        self.data=data
        self.next=None

class LinkedList():
    def __init__(self):
        self.head=None

    def addNode(self,data):
        newNode=Node(data)
        
        if self.head is None:
            self.head = newNode
            return
        temp = self.head
        while temp.next:
            temp=temp.next
            temp.next=newNode
            
    def listView(self):
        temp=self.head
        while temp:
            print(temp.data,end="->")
            temp=temp.next


if __name__=='__main__':
    x = np.random.uniform(0.0,1.0,16)
    print(x)

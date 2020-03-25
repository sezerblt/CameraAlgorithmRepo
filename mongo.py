import pymongo

myclient=pymongo.MongoClient("mongodb://localhost:27017/")
mydb=myclient["mydatabase"]
mycol=mydb["customers"]

mylist=[{"name":"Yusuf", "address":"istanbul/Basaksehir"},
        {"name":"Fatih", "address":"Erzurum/Horasan"},
        {"name":"galip", "address":"istanbul/Fatih"},
        {"name":"Ayhan", "address":"istanbul/Maltepe"},
        {"name":"Fuat",  "address":"Antalya/Muratpasa"},
        {"name":"Serdar","address":"Ankara/Yenimahalle"}
        ]
x=mycol.insert_many(mylist)
y=mycol.find_one()
y_all=mycol.find()

print("veritabanlari: ",myclient.list_database_names())
print("Kolleksiyonlar: ",mydb.list_collection_names())

print(x.inserted_ids)
print(y)
for x in mycol.find({},{"name":"Hasan"}):
  print(x)

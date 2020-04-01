"""
Baglantılı liste, her biri bir Referans ve bir Degerden oluşan
düğüm koleksiyonudur. Düğümler birbirine bir
referanslarını kullanarak dizi. Bağlantılı listeler,
listeler gibi daha karmaşık veri yapılarını uygulamak için kullanılabilir.
yığınlar, kuyruklar ve ilişkilendirilebilir diziler.
"""
class Node:
    def __init__(self,value):
        self.data = value
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self,value):
        self.data = value

    def setNext(self,value):
        self.next = value

class LinkedList:
    def __init__(self):
        #liste basi
        self.head = None

    def isEmpty(self):
        """eger liste boş ise"""
        return self.head is None

    def add(self,item):
        """listeye dugum ekle"""
        #dugum olustur
        node = Node(item)
        #dugum next degeri bir önceki dügüm olsun
        node.setNext(self.head) #next degeri -None- oldu
        #dugüm basa eklendi
        self.head = node

    def size(self):
        """listenin uzunlugu dondur"""
        count = 0
        current = self.head #liste bos degilmi kontroloru
        while current is not None: #ccurrent!=None
            count += 1
            current = current.getNext() #sonraki data git
        return count

    def search(self,item):
        current = self.head
        found = False
        # current degeri None degilse ve
        # istenilen deger bulunmadıysa
        while (current is not None) and (not found):
            # current datası item ile eslestiyse 
            if current.getData() is item:
                # bulundu mode aktif
                found = True
            else:
                # sonraki data git
                current = current.getNext()
        return found

    def remove(self,item):
        """item listeden sil"""
        current = self.head
        previous = None
        found = False
        while (current is not None) and (not found):
            if current.getData() is item:
                found = True
            else:
                previous = current
                current = current.getNext()
        if found:
            if previous is None:
                self.head=curent.getNext()
            else:
                previous.setNext(current.getNext())
        else:
            raise ValueError
            print(item," not found")

    def insert(self,position,item):
        #index degeri buyukse listenin boyutundan büyükse
        if position>size()-1:
            raise IndexError
            print("index hatasi")
        #current basa sar
        current = self.head
        previous = None
        pos = 0
        #index o ise add metodu ile basa al
        if (position==0):
            self.add(item)
        else:
            #duğm olustur
            node = Node(item)
            #listeyi gez
            while pos<position:
                pos += 1
                #current ile pre aynı dügümü 
                previous = current
                #current sonraki dügümü tutsun
                current = current.getNext()
            #prev nexti olusturulan dugum olsun
            previous.setNext(node)
            #dügümün nexti current tuttugu dügüm 
            node.setNext(current)

    def index(self,item):
        current=self.head
        pos=0
        found = False
        while (current is not None) and (not Found):
            if current.getData() is item:
                found = True
            else:
                curent = current.getNext()
                pos +=1
        if found:
            print("item eslesti")
        else:
            pos = None
        return pos

    def pop(self,position=None):
        if position > self.size():
            print(" hata")
            raise IndexError
        current =self.head
        ret=None
        if position is None:
            ret = current.getData()
            self.head=current.getNext()
        else:
            pos=0
            previous = None
            while pos<position:
                previous = current
                current = current.getNext()
                pos += 1
                ret = curent.getData()
            previous.setNext(current.getNext())
        print(ret)
        return ret

    def appendItem(self,item):
        #current bas dugumu tut
        current = self.head
        previous = None
        pos=0
        lenght = self.size()
        #index liste boyutunca
        while pos<lenght:
            #previous current isaretci tnı dugum tutsun
            previous = current
            #current kendi next dugumu tutsun
            current = current.getNext()
            pos += 1
        node = Node(item)
        if previous is None:
            node.setNext(current)
            self.head = node
        else:
            previous.setNext(node)

    def printList(self):
        current=self.head
        while current is not None:
            print(current.getData())
            current = current.getNext()
            
   
        

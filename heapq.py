import heapq

score = [15,63,5,45,63,98,78,25,25,35,47,84,24]
#en buyuk degrleri alaır
#parametre degerine göre
maxFour = heapq.nlargest(4,score)
minFour = heapq.nsmallest(3,score)
sort_score=heapq.heapify(score)
pop_score = heapq.heappop(score)
print(minFour)
print(maxFour)
print("ify: ",sort_score)
print("pop: ",pop_score)
print("====================")

artist=[
    {'firstname':'Taylor','lastname':'Swifth','album':['red'],'age':34},
    {'firstname':'Edd ', 'lastname':'Sheran',  'album':['my live'],'age':28},
    {'firstname':'sia','lastname':'sia','album':['cheap thrill'],'age':45},
    {'firstname':'sezen','lastname':'aksu','album':['ah istanbul'],'age':55},
    {'firstname':'sebnem','lastname':'ferah','album':['mayın tarlası'],'age':44},
    {'firstname':'tarkan','lastname':'tevt','album':['dudu dudu'],'age':44},
    {'firstname':'tıvıro','lastname':'ismail','album':['aaaa aaa'],'age':65}
   ]
oldest=heapq.nlargest(2,artist,key=lambda s: s['age'])
print(oldest)


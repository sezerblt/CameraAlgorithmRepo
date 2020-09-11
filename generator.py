import requests
import json

api_url="https://api.exchangeratesapi.io/latest?base="

sal_exc=input("satilan doviz:")
get_exc=input("alinan doviz:")
amount = input(f"ne kadar {sal_exc} bozurulacak: ")
amount =int(amount)

result = requests.get(api_url+sal_exc)
result =json.loads(result.text)

views="1 {0}={1} {2}".format(sal_exc,result['rates'][get_exc],get_exc)
fone ="-> {0} {1}={2} {3}".format(amount,sal_exc,amount*result['rates'][get_exc],get_exc)

print(views)
print(fone)
 


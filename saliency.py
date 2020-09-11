class Car(object):
    wheel=4
    mark="any car"
    coin=100
    @classmethod
    def getCountWheel(cls):
        return cls.wheel

    @classmethod
    def setMark(cls,mark1):
        cls.mark==mark1

    @classmethod
    def setCoin(cls,money):
        return cls.coin+money


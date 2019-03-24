
#正向最大匹配法
class MM(object):

   def __init__(self):
       self.max_size = 3


   def cut(self, text):
        result=[]
        index=0
        text_len = len(text)
        dic = ['研究','研究生','生命','命','的','起源']
        while text_len > index:
            for size in range(self.max_size+index,index,-1):
                piece = text[index:size]
                if piece in dic:
                    index = size-1
                    break
            index += 1
            result.append(piece)
        return result

# 逆向最大匹配
class RMM(object):
    def __init__(self):
        self.max_size = 3

    def cut(self, text):
        result = []
        index = len(text)
        dic = ['研究', '研究生', '生命', '命', '的', '起源']
        while index > 0:
            for size in range(index - self.max_size, index):
                piece = text[size:index]
                if piece in dic:
                    index = size + 1
                    break
            index -= 1
            result.append(piece)
        result.reverse()
        return result


if __name__ == '__main__':
    text = '研究生命的起源'
    count1 = 0
    count2 = 0
    First = MM()
    Second = RMM()
    a = First.cut(text)
    b = Second.cut(text)
    print("正向最大匹配结果",a)
    print("反向最大匹配结果",b)
    if a == b:
        print(a)
    lena = len(a)
    lenb = len(b)
    if lena == lenb:
        for DY1 in a:
            if len(DY1) == 5:
                count1 = count1 + 1
        for DY2 in b:
            if len(DY2) == 5:
                count2 = count2 + 1
        if count1 > count2:
            print(b)
        else:
            print(a)
    if lena > lenb:
        print(b)
    if lena < lenb:
        print(a)




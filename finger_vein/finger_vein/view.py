from django.shortcuts import render, HttpResponse
import sys
sys.path.append('..')
import recogAndMatch as ram

def start(request):
    return render(request, 'match.html')

def getMatch(request):

    num1=request.GET['num1']
    num2=request.GET['num2']
    print(num1)
    print(num2)
    print(type(num2))

    LBP_score,PBBM_score,dis,ismatch=ram.match(num1,num2)
    print(ismatch)
    if ismatch==True:
        data="选择的两幅图像来自同一个人"
    else:
        data="选择的两幅图像不匹配"
    c={
        'LBP_score':LBP_score,
        'PBBM_score':PBBM_score,
        'dis':dis,
        'number1':num1,
        'number2':num2,
        'data':data,
    }
    return render(request,'result.html',c)
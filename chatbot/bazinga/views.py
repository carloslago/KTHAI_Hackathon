from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .functions import predict_input
# Create your views here.

def home(request):
    return render(request, "index.html")

@csrf_exempt
def predict(request):
    context = {}
    if request.method == 'POST':
        input = request.POST.get('message')
        context['response'] = predict_input(input)
        context['question'] = input
    return render(request, "predict.html", context)
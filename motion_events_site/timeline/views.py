from django.shortcuts import render

from .models import Class, Instance, Event

# Create your views here.

def index(request):
    instance_list = Instance.objects.all()



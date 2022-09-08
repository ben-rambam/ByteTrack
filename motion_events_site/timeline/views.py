from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render

from .models import TrackableClass, Trackable, Event

# Create your views here.

def index(request):
    trackable_list = Trackable.objects.all()
    event_list = Event.objects.order_by('date')

    context = {
            'trackable_list': trackable_list,
            'event_list': event_list,
            }
    return render(request, 'timeline/index.html', context)


